#!/usr/bin/env bun
/**
 * Epoch-2 devnet bootstrap for the pc-wallet-kit deployment.
 *
 * Creates, against already-deployed pc-token / pc-swap programs:
 *   mock-USDC + mock-DOGE SPL mints (deployer is mint authority, for the faucet),
 *   pc-mints, vaults and vault token accounts, the deployer's pc-accounts,
 *   a USDC/DOGE pool seeded from the deployer's wrapped balance.
 *
 * Resumable: every step checks the chain (or the state file) first, so a rerun
 * continues after a dropped executor event. Writes the devnet.json fragment to
 * the state file's `devnet` key.
 *
 * Usage: bun bootstrap-epoch2.ts <ENCRYPT_ID> <PC_TOKEN_ID> <PC_SWAP_ID>
 * Env:   KEYPAIR_PATH (deployer), STATE_PATH, AUTH_DIR, RPC_URL, GRPC_URL, SEED_TOKENS
 */
import { Connection, Keypair, PublicKey, SystemProgram, TransactionInstruction } from "@solana/web3.js";
import * as fs from "fs";
import * as path from "path";
import type { EncryptAccounts } from "../../_shared/encrypt-setup.ts";
import { log, ok, sendTx, pda, mockCiphertext } from "../../_shared/helpers.ts";
import { createEncryptWebClient, Chain } from "../../../clients/typescript/src/grpc-web.ts";
import { type SwapContext, deriveSwapPdas, derivePoolPda, deriveLpPda, createPoolIx, createLpPositionIx, addLiquidityIx } from "./instructions.ts";
import {
  type PcTokenContext, derivePcTokenPdas, deriveMintPda, deriveAccountPda, deriveVaultPda,
  initializeMintIx, initializeAccountIx, initializeVaultIx, wrapIx,
} from "../../pc-token/e2e/instructions.ts";
import { createSplMint, splMintToIx, TOKEN_PROGRAM_ID } from "../../pc-token/e2e/spl-helpers.ts";

const RPC = process.env.RPC_URL ?? "https://api.devnet.solana.com";
const HOME = process.env.HOME!;
const DECIMALS = Number(process.env.DECIMALS ?? "2");
const STATE_PATH = process.env.STATE_PATH ?? `${HOME}/.config/solana/epoch2-bootstrap-state-d${DECIMALS}.json`;
const AUTH_DIR = process.env.AUTH_DIR ?? `${HOME}/.config/solana`;
// 2 decimals by default: pc-swap's u64 curve math overflows with 6-decimal reserves of
// this size (receipt * 997 * reserve_out must stay below 2^64).
const TOKENS = (n: number) => BigInt(n) * 10n ** BigInt(DECIMALS);
const SEED = Number(process.env.SEED_TOKENS ?? "100000");     // per side, into the pool
const SUPPLY = 1_000_000;                                       // per side, minted to deployer
const FHE64 = 4;
const ATA_PROGRAM = new PublicKey("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL");

const [eArg, tArg, sArg] = process.argv.slice(2);
if (!eArg || !tArg || !sArg) { console.error("Usage: bun bootstrap-epoch2.ts <ENCRYPT_ID> <PC_TOKEN_ID> <PC_SWAP_ID>"); process.exit(1); }
const EP = new PublicKey(eArg), TP = new PublicKey(tArg), SP = new PublicKey(sArg);
const conn = new Connection(RPC, "confirmed");
const loadKp = (p: string) => Keypair.fromSecretKey(Uint8Array.from(JSON.parse(fs.readFileSync(p, "utf-8"))));
const payer = loadKp(process.env.KEYPAIR_PATH ?? `${HOME}/.config/solana/id.json`);

type State = Record<string, any>;
const state: State = fs.existsSync(STATE_PATH) ? JSON.parse(fs.readFileSync(STATE_PATH, "utf-8")) : {};
const save = () => fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));
const pk = (s: string) => new PublicKey(s);
const exists = async (a: PublicKey) => !!(await conn.getAccountInfo(a));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Load a keypair file, creating it (0600) if missing. Never logs secrets. */
function loadOrCreateKp(name: string): Keypair {
  const p = path.join(AUTH_DIR, name);
  if (fs.existsSync(p)) return loadKp(p);
  const kp = Keypair.generate();
  fs.writeFileSync(p, JSON.stringify(Array.from(kp.secretKey)), { mode: 0o600 });
  return kp;
}

function ataAddress(mint: PublicKey, owner: PublicKey): PublicKey {
  return PublicKey.findProgramAddressSync([owner.toBuffer(), TOKEN_PROGRAM_ID.toBuffer(), mint.toBuffer()], ATA_PROGRAM)[0];
}
function createAtaIdempotentIx(mint: PublicKey, owner: PublicKey): TransactionInstruction {
  return new TransactionInstruction({
    programId: ATA_PROGRAM, data: Buffer.from([1]),
    keys: [
      { pubkey: payer.publicKey, isSigner: true, isWritable: true },
      { pubkey: ataAddress(mint, owner), isSigner: false, isWritable: true },
      { pubkey: owner, isSigner: false, isWritable: false },
      { pubkey: mint, isSigner: false, isWritable: false },
      { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
      { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    ],
  });
}

async function digest(ct: PublicKey): Promise<string | null> {
  const i = await conn.getAccountInfo(ct);
  return i ? Buffer.from(i.data.subarray(2, 34)).toString("hex") : null;
}
async function status(ct: PublicKey): Promise<number | null> {
  const i = await conn.getAccountInfo(ct);
  return i && i.data.length >= 100 ? i.data[99] : null;
}
/**
 * Settled = status byte 1 (Verified) and, for an existing ciphertext, a digest different from `before`.
 * The digest alone is not enough: Encrypt rewrites it (and sets status 0, Pending) when the
 * transaction lands, before the executor has committed anything.
 */
async function waitCommitted(ct: PublicKey, before: string | null, label: string, timeoutMs = 180_000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const d = await digest(ct); const s = await status(ct);
    if (s === 1 && (before === null || (d !== null && d !== before))) { ok(`${label} committed (${Math.round((Date.now() - t0) / 1000)}s)`); return; }
    await sleep(1500);
  }
  throw new Error(`${label}: executor did not commit ${ct.toBase58()} within ${timeoutMs / 1000}s (dropped event?). Rerun to continue.`);
}

async function main() {
  console.log(`\nEpoch-2 bootstrap — payer ${payer.publicKey.toBase58()}, ${(await conn.getBalance(payer.publicKey)) / 1e9} SOL`);
  console.log(`state: ${STATE_PATH}\n`);
  const grpc = createEncryptWebClient(process.env.GRPC_URL ?? "https://pre-alpha-dev-1.encrypt.ika-network.net:443");

  // ── Encrypt accounts + deposit ──
  const [cfgPda] = pda([Buffer.from("encrypt_config")], EP);
  const [evtAuth] = pda([Buffer.from("__event_authority")], EP);
  const [depPda, depBump] = pda([Buffer.from("encrypt_deposit"), payer.publicKey.toBuffer()], EP);
  const nk = Buffer.alloc(32, 0x55);
  const [nkPda] = pda([Buffer.from("network_encryption_key"), nk], EP);
  if (!(await exists(depPda))) {
    log("0", "Creating Encrypt deposit");
    const ci = await conn.getAccountInfo(cfgPda); if (!ci) throw new Error("Encrypt config missing");
    const ev = new PublicKey((ci.data as Buffer).subarray(100, 132));
    const vp = ev.equals(PublicKey.default) ? payer.publicKey : ev;
    const dd = Buffer.alloc(18); dd[0] = 14; dd[1] = depBump;
    await sendTx(conn, payer, [new TransactionInstruction({ programId: EP, data: dd, keys: [
      { pubkey: depPda, isSigner: false, isWritable: true }, { pubkey: cfgPda, isSigner: false, isWritable: false },
      { pubkey: payer.publicKey, isSigner: true, isWritable: false }, { pubkey: payer.publicKey, isSigner: true, isWritable: true },
      { pubkey: payer.publicKey, isSigner: true, isWritable: true }, { pubkey: vp, isSigner: vp.equals(payer.publicKey), isWritable: true },
      { pubkey: PublicKey.default, isSigner: false, isWritable: false }, { pubkey: PublicKey.default, isSigner: false, isWritable: false }] })]);
  }
  ok("Encrypt deposit ready");

  const enc: EncryptAccounts = { encryptProgram: EP, configPda: cfgPda, eventAuthority: evtAuth, depositPda: depPda, networkKeyPda: nkPda, networkKey: nk };
  const tPdas = derivePcTokenPdas(TP, payer.publicKey);
  const sPdas = deriveSwapPdas(SP, TP);
  const tCtx: PcTokenContext = { programId: TP, enc, payer: payer.publicKey, cpiAuthority: tPdas.cpiAuthority, cpiBump: tPdas.cpiBump };
  const sCtx: SwapContext = { programId: SP, pcTokenProgramId: TP, enc, payer: payer.publicKey,
    cpiAuthority: sPdas.cpiAuthority, cpiBump: sPdas.cpiBump, pcTokenCpiAuthority: sPdas.pcTokenCpiAuthority, pcTokenCpiBump: sPdas.pcTokenCpiBump };

  const sides = ["USDC", "DOGE"] as const;
  state.sides ??= {};
  for (const side of sides) {
    const st = (state.sides[side] ??= {});
    log("1", `${side}: SPL mint, pc-mint, vault`);
    if (!st.splMint) {
      st.splMint = (await createSplMint(conn, payer, DECIMALS, payer.publicKey)).publicKey.toBase58(); save();
    }
    const splMint = pk(st.splMint);
    const payerAta = ataAddress(splMint, payer.publicKey);
    if (!st.supplyMinted) {
      await sendTx(conn, payer, [createAtaIdempotentIx(splMint, payer.publicKey), splMintToIx(splMint, payerAta, payer.publicKey, TOKENS(SUPPLY))]);
      st.supplyMinted = true; save();
    }
    const auth = loadOrCreateKp(`epoch2-pc${side.toLowerCase()}-d${DECIMALS}-mint-authority.json`);
    const [pcMint, pcMintBump] = deriveMintPda(TP, auth.publicKey);
    if (!(await exists(pcMint))) await sendTx(conn, payer, [initializeMintIx(tCtx, pcMint, pcMintBump, DECIMALS, auth.publicKey)], [auth]);
    const [vault, vaultBump] = deriveVaultPda(TP, pcMint);
    if (!(await exists(vault))) await sendTx(conn, payer, [initializeVaultIx(tCtx, vault, vaultBump, pcMint, splMint, auth.publicKey)], [auth]);
    const vaultAta = ataAddress(splMint, vault);
    if (!(await exists(vaultAta))) await sendTx(conn, payer, [createAtaIdempotentIx(splMint, vault)]);
    Object.assign(st, { pcMint: pcMint.toBase58(), pcMintAuthority: auth.publicKey.toBase58(), vault: vault.toBase58(), vaultAta: vaultAta.toBase58() });
    save();

    // Deployer's pc-account (pool seeding comes from the deployer's wrapped balance).
    const [acct, acctBump] = deriveAccountPda(TP, pcMint, payer.publicKey);
    if (!(await exists(acct))) {
      const balKp = Keypair.generate();
      await sendTx(conn, payer, [initializeAccountIx(tCtx, acct, acctBump, pcMint, payer.publicKey, balKp.publicKey)], [balKp]);
      await waitCommitted(balKp.publicKey, null, `${side} deployer balance init`);
      await sleep(5000); // guide §3: keep setup and the first wrap apart
    }
    const balCt = new PublicKey((await conn.getAccountInfo(acct))!.data.subarray(64, 96));
    st.deployerAccount = acct.toBase58(); st.deployerBalanceCt = balCt.toBase58(); save();

    if (!st.wrapped) {
      log("2", `${side}: wrapping ${SEED} into the deployer's pc-account (epoch-2 wrap: fresh amount keypair)`);
      const before = await digest(balCt);
      const amtKp = Keypair.generate();
      await sendTx(conn, payer, [wrapIx(tCtx, vault, acct, payerAta, vaultAta, balCt, amtKp.publicKey, payer.publicKey, TOKENS(SEED))], [amtKp]);
      st.wrapped = true; save();
      await waitCommitted(balCt, before, `${side} wrap`);
    }
  }

  // ── Pool ──
  const usd = state.sides.USDC, doge = state.sides.DOGE;
  const mintA = pk(usd.pcMint), mintB = pk(doge.pcMint);
  const [pool, poolBump] = derivePoolPda(SP, mintA, mintB);
  const [vaultA, vaultABump] = deriveAccountPda(TP, mintA, pool);
  const [vaultB, vaultBBump] = deriveAccountPda(TP, mintB, pool);
  state.pool ??= {};
  if (!(await exists(pool))) {
    log("3", "Creating pool (pc-swap CPIs pc-token::initialize_account for both vaults)");
    const kps = { vaBal: Keypair.generate(), vbBal: Keypair.generate(), resA: Keypair.generate(), resB: Keypair.generate(), supply: Keypair.generate() };
    await sendTx(conn, payer, [createPoolIx(sCtx, pool, poolBump, mintA, mintB, vaultA, vaultABump, vaultB, vaultBBump,
      kps.vaBal.publicKey, kps.vbBal.publicKey, kps.resA.publicKey, kps.resB.publicKey, kps.supply.publicKey)],
      [kps.vaBal, kps.vbBal, kps.resA, kps.resB, kps.supply]);
    Object.assign(state.pool, { address: pool.toBase58(), vaultAccountA: vaultA.toBase58(), vaultAccountB: vaultB.toBase58(),
      vaultBalanceCtA: kps.vaBal.publicKey.toBase58(), vaultBalanceCtB: kps.vbBal.publicKey.toBase58(),
      reserveCtA: kps.resA.publicKey.toBase58(), reserveCtB: kps.resB.publicKey.toBase58(), totalSupplyCt: kps.supply.publicKey.toBase58() });
    save();
    for (const k of ["reserveCtA", "reserveCtB", "totalSupplyCt"]) await waitCommitted(pk(state.pool[k]), null, `pool ${k} init`);
  }
  const P = state.pool;
  const [lpPda] = deriveLpPda(SP, pool, payer.publicKey);
  if (!(await exists(lpPda))) {
    log("3", "Creating deployer LP position");
    const lpKp = Keypair.generate();
    await sendTx(conn, payer, [createLpPositionIx(sCtx, pool, payer.publicKey, lpKp.publicKey)], [lpKp]);
    P.lpBalanceCt = lpKp.publicKey.toBase58(); save();
    await waitCommitted(lpKp.publicKey, null, "LP balance init");
  }

  if (!P.seeded) {
    log("4", `AddLiquidity ${SEED} : ${SEED}`);
    const input = async (v: bigint) => {
      const ciphertextIdentifiers = await grpc.createInput({ chain: Chain.SOLANA,
        inputs: [{ ciphertextBytes: mockCiphertext(v, FHE64), fheType: FHE64 }], authorized: SP.toBytes(), networkEncryptionPublicKey: nk });
      return new PublicKey(ciphertextIdentifiers[0]);
    };
    const amtA = await input(TOKENS(SEED)), amtB = await input(TOKENS(SEED));
    for (const ct of [amtA, amtB]) { const t0 = Date.now(); while ((await status(ct)) !== 1) { if (Date.now() - t0 > 90_000) throw new Error("createInput not verified"); await sleep(1000); } }
    const before = await digest(pk(P.reserveCtA));
    const rA = Keypair.generate(), rB = Keypair.generate();
    await sendTx(conn, payer, [addLiquidityIx(sCtx, pool, pk(usd.deployerAccount), pk(doge.deployerAccount), vaultA, vaultB,
      pk(usd.deployerBalanceCt), pk(doge.deployerBalanceCt), pk(P.vaultBalanceCtA), pk(P.vaultBalanceCtB),
      pk(P.reserveCtA), pk(P.reserveCtB), pk(P.totalSupplyCt), amtA, amtB, pk(P.lpBalanceCt), rA.publicKey, rB.publicKey)], [rA, rB]);
    P.seeded = true; save();
    await waitCommitted(pk(P.reserveCtA), before, "reserve A after AddLiquidity");
    await waitCommitted(pk(P.reserveCtB), null, "reserve B after AddLiquidity");
  }

  state.devnet = {
    programs: { encrypt: EP.toBase58(), pcToken: TP.toBase58(), pcSwap: SP.toBase58() },
    sides: Object.fromEntries(sides.map((s) => [s, { decimals: DECIMALS, splMint: state.sides[s].splMint, pcMint: state.sides[s].pcMint,
      pcMintAuthority: state.sides[s].pcMintAuthority, vault: state.sides[s].vault, vaultAta: state.sides[s].vaultAta }])),
    pool: { address: P.address, sideA: "USDC", sideB: "DOGE", vaultAccountA: P.vaultAccountA, vaultAccountB: P.vaultAccountB,
      vaultBalanceCtA: P.vaultBalanceCtA, vaultBalanceCtB: P.vaultBalanceCtB, reserveCtA: P.reserveCtA, reserveCtB: P.reserveCtB },
  };
  save();
  console.log("\nDone. devnet.json fragment:\n" + JSON.stringify(state.devnet, null, 2));
}

main().catch((e) => { console.error("Error:", e?.message ?? e); process.exit(1); });
