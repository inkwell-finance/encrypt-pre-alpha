// Copyright (c) 2026 Inkwell Finance
// SPDX-License-Identifier: BSD-3-Clause-Clear

//! Epoch-2 regression tests: each test reproduces one hole found in the
//! pre-epoch-2 program (reported by an integrating wallet and by our own
//! review) and checks that the hardened program refuses it, plus the honest
//! paths that the checks must keep working (wrap, unwrap end to end).
//!
//! Runs against the `localnet` build (`target/deploy-localnet/pc_token.so`),
//! the Encrypt program binary in `bin/`, and LiteSVM's built-in SPL Token.

use std::str::FromStr;

use encrypt_dsl::prelude::encrypt_fn;
use encrypt_dev::runtime::SolanaRuntime;
use encrypt_service::store::CiphertextStore;
use encrypt_solana_test::litesvm::EncryptTestContext;
use encrypt_types::encrypted::{EUint64, Uint64};
use encrypt_compute::engine::ComputeEngine;
use encrypt_types::types::FheType;
use solana_sdk::instruction::{AccountMeta, Instruction};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::transaction::Transaction;

#[encrypt_fn]
fn mint_to_graph(balance: EUint64, amount: EUint64) -> EUint64 {
    balance + amount
}

#[encrypt_fn]
fn unwrap_burn_graph(balance: EUint64, amount: EUint64) -> (EUint64, EUint64) {
    let s = balance >= amount;
    let new_balance = if s { balance - amount } else { balance };
    let burned = if s { amount } else { amount - amount };
    (new_balance, burned)
}

const PROGRAM_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../../target/deploy-localnet/pc_token.so"
);

const SYSTEM_PROGRAM: Pubkey = Pubkey::new_from_array([0u8; 32]);

fn spl_token_id() -> Pubkey {
    Pubkey::from_str("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA").unwrap()
}

// Custom error codes (pc-token `err` module).
const FOREIGN_ENCRYPT: u32 = 0x100;
const BAD_VAULT: u32 = 0x102;
const NOT_FRESH: u32 = 0x105;
const RECEIPT_MISMATCH: u32 = 0x106;
const DECRYPT_ALREADY_REQUESTED: u32 = 0x107;
const ALIASED: u32 = 0x109;
const NOT_MINT_AUTHORITY: u32 = 0x10C;

// ── Harness ──

struct Env {
    ctx: EncryptTestContext,
    program_id: Pubkey,
    cpi_authority: Pubkey,
    cpi_bump: u8,
}

/// One pc mint with its vault and backing SPL mint.
struct Side {
    mint: Pubkey,
    authority: Keypair,
    spl_mint: Pubkey,
    vault: Pubkey,
    vault_bump: u8,
    vault_ata: Pubkey,
}

struct Holder {
    owner: Keypair,
    pda: Pubkey,
    balance_ct: Pubkey,
    ata: Pubkey,
}

impl Env {
    fn new() -> Self {
        let mut ctx = EncryptTestContext::new_default();
        let program_id = ctx.deploy_program(PROGRAM_PATH);
        let (cpi_authority, cpi_bump) = ctx.cpi_authority_for(&program_id);
        Env { ctx, program_id, cpi_authority, cpi_bump }
    }

    fn try_send(&mut self, ixs: &[Instruction], signers: &[&Keypair]) -> Result<(), String> {
        let blockhash = self.ctx.latest_blockhash();
        let payer = self.ctx.payer().insecure_clone();
        let mut all: Vec<&Keypair> = vec![&payer];
        all.extend_from_slice(signers);
        let tx = Transaction::new_signed_with_payer(ixs, Some(&payer.pubkey()), &all, blockhash);
        self.ctx
            .harness_mut()
            .tx_builder_mut()
            .runtime_mut()
            .send_transaction(&tx)
            .map_err(|e| format!("{e:?}"))
    }

    /// Send, and return the plaintext of the Encrypt `create_plaintext` CPI
    /// (discriminator 2, fhe type 4 = u64) that creates `ct`.
    fn send_capturing_plaintext(&mut self, ixs: &[Instruction], signers: &[&Keypair], ct: &Pubkey) -> u64 {
        let blockhash = self.ctx.latest_blockhash();
        let payer = self.ctx.payer().insecure_clone();
        let mut all: Vec<&Keypair> = vec![&payer];
        all.extend_from_slice(signers);
        let tx = Transaction::new_signed_with_payer(ixs, Some(&payer.pubkey()), &all, blockhash);
        let keys: Vec<Pubkey> = tx.message.account_keys.clone();
        let bytes = bincode::serialize(&tx).unwrap();
        let vtx: solana_transaction::versioned::VersionedTransaction = bincode::deserialize(&bytes).unwrap();
        let meta = self
            .ctx
            .harness_mut()
            .tx_builder_mut()
            .runtime_mut()
            .inner_mut()
            .send_transaction(vtx)
            .map_err(|e| format!("{e:?}"))
            .expect("transaction failed");
        let encrypt = *self.ctx.program_id();
        let mut found = None;
        for inner in meta.inner_instructions.iter().flatten() {
            let ix = &inner.instruction;
            let program = keys[ix.program_id_index as usize];
            let names_ct = ix.accounts.iter().any(|i| keys[*i as usize] == *ct);
            if program == encrypt && names_ct && ix.data.len() == 10 && ix.data[0] == 2 && ix.data[1] == 4 {
                found = Some(u64::from_le_bytes(ix.data[2..10].try_into().unwrap()));
            }
        }
        found.expect("no create_plaintext CPI for the amount ciphertext")
    }

    fn send(&mut self, ixs: &[Instruction], signers: &[&Keypair]) {
        self.try_send(ixs, signers).expect("transaction failed");
    }

    /// The Encrypt account tail shared by every pc-token instruction.
    fn tail(&self, payer: &Pubkey) -> Vec<AccountMeta> {
        vec![
            AccountMeta::new_readonly(*self.ctx.program_id(), false),
            AccountMeta::new(*self.ctx.config_pda(), false),
            AccountMeta::new(*self.ctx.deposit_pda(), false),
            AccountMeta::new_readonly(self.cpi_authority, false),
            AccountMeta::new_readonly(self.program_id, false),
            AccountMeta::new_readonly(*self.ctx.network_encryption_key_pda(), false),
            AccountMeta::new(*payer, true),
            AccountMeta::new_readonly(*self.ctx.event_authority(), false),
            AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
        ]
    }

    fn ix(&self, data: Vec<u8>, accounts: Vec<AccountMeta>) -> Instruction {
        Instruction::new_with_bytes(self.program_id, &data, accounts)
    }

    // ── SPL accounts, written directly ──

    fn spl_mint(&mut self, authority: &Pubkey) -> Pubkey {
        let key = Pubkey::new_unique();
        let mut d = vec![0u8; 82];
        d[0..4].copy_from_slice(&[1, 0, 0, 0]);
        d[4..36].copy_from_slice(authority.as_ref());
        d[36..44].copy_from_slice(&u64::MAX.to_le_bytes()); // supply (irrelevant)
        d[44] = 6; // decimals
        d[45] = 1; // is_initialized
        let lamports = self.ctx.minimum_balance(82);
        self.ctx.set_account(&key, d, &spl_token_id(), lamports);
        key
    }

    fn spl_account(&mut self, mint: &Pubkey, owner: &Pubkey, amount: u64) -> Pubkey {
        let key = Pubkey::new_unique();
        let mut d = vec![0u8; 165];
        d[0..32].copy_from_slice(mint.as_ref());
        d[32..64].copy_from_slice(owner.as_ref());
        d[64..72].copy_from_slice(&amount.to_le_bytes());
        d[108] = 1; // AccountState::Initialized
        let lamports = self.ctx.minimum_balance(165);
        self.ctx.set_account(&key, d, &spl_token_id(), lamports);
        key
    }

    fn spl_amount(&self, key: &Pubkey) -> u64 {
        let d = self.ctx.get_account_data(key).unwrap();
        u64::from_le_bytes(d[64..72].try_into().unwrap())
    }

    // ── pc-token setup ──

    fn side(&mut self) -> Side {
        let authority = self.ctx.new_funded_keypair();
        let (mint, mint_bump) =
            Pubkey::find_program_address(&[b"pc_mint", authority.pubkey().as_ref()], &self.program_id);
        let mut data = vec![0u8, mint_bump, 6];
        data.extend_from_slice(authority.pubkey().as_ref());
        data.push(0);
        let payer = self.ctx.payer().pubkey();
        let ix = self.ix(
            data,
            vec![
                AccountMeta::new(mint, false),
                AccountMeta::new_readonly(authority.pubkey(), true),
                AccountMeta::new(payer, true),
                AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
            ],
        );
        self.send(&[ix], &[&authority]);

        let spl_mint = self.spl_mint(&authority.pubkey());
        let (vault, vault_bump) =
            Pubkey::find_program_address(&[b"pc_vault", mint.as_ref()], &self.program_id);
        let ix = self.init_vault_ix(&vault, vault_bump, &mint, &spl_mint, &authority.pubkey());
        self.send(&[ix], &[&authority]);
        let vault_ata = self.spl_account(&spl_mint, &vault, 0);
        Side { mint, authority, spl_mint, vault, vault_bump, vault_ata }
    }

    fn init_vault_ix(&self, vault: &Pubkey, bump: u8, mint: &Pubkey, spl_mint: &Pubkey, authority: &Pubkey) -> Instruction {
        let payer = self.ctx.payer().pubkey();
        self.ix(
            vec![23, bump],
            vec![
                AccountMeta::new(*vault, false),
                AccountMeta::new_readonly(*mint, false),
                AccountMeta::new_readonly(*spl_mint, false),
                AccountMeta::new(payer, true),
                AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
                AccountMeta::new_readonly(*authority, true),
            ],
        )
    }

    fn holder(&mut self, side: &Side, spl_amount: u64) -> Holder {
        let owner = self.ctx.new_funded_keypair();
        let (pda, bump) = Pubkey::find_program_address(
            &[b"pc_account", side.mint.as_ref(), owner.pubkey().as_ref()],
            &self.program_id,
        );
        let bal = Keypair::new();
        let payer = self.ctx.payer().pubkey();
        let mut accts = vec![
            AccountMeta::new(pda, false),
            AccountMeta::new_readonly(side.mint, false),
            AccountMeta::new_readonly(owner.pubkey(), false),
            AccountMeta::new(bal.pubkey(), true),
        ];
        accts.extend(self.tail(&payer));
        let ix = self.ix(vec![1, bump, self.cpi_bump], accts);
        self.send(&[ix], &[&bal]);
        self.ctx.register_ciphertext(&bal.pubkey());
        let ata = self.spl_account(&side.spl_mint, &owner.pubkey(), spl_amount);
        Holder { owner, pda, balance_ct: bal.pubkey(), ata }
    }

    fn wrap_ix(&self, side_vault: &Pubkey, side_vault_ata: &Pubkey, h: &Holder, amt_ct: &Pubkey, amt_ct_signer: bool, amount: u64) -> Instruction {
        let mut data = vec![30, self.cpi_bump];
        data.extend_from_slice(&amount.to_le_bytes());
        let mut accts = vec![
            AccountMeta::new_readonly(*side_vault, false),
            AccountMeta::new_readonly(h.pda, false),
            AccountMeta::new(h.ata, false),
            AccountMeta::new(*side_vault_ata, false),
            AccountMeta::new(h.balance_ct, false),
            AccountMeta::new(*amt_ct, amt_ct_signer),
        ];
        accts.extend(self.tail(&h.owner.pubkey()));
        accts.push(AccountMeta::new_readonly(spl_token_id(), false));
        self.ix(data, accts)
    }

    /// Wrap honestly and settle it the way the executor would.
    fn wrap(&mut self, side: &Side, h: &Holder, amount: u64) {
        let amt = Keypair::new();
        let ix = self.wrap_ix(&side.vault, &side.vault_ata, h, &amt.pubkey(), true, amount);
        let committed = self.send_capturing_plaintext(&[ix], &[&h.owner, &amt], &amt.pubkey());
        assert_eq!(committed, amount, "the program must create amt_ct from the SPL amount");
        // What the executor does with the CiphertextCreated event: commit the
        // plaintext carried by the CPI (not a value the client chose).
        let digest = self
            .ctx
            .harness_mut()
            .engine_mut()
            .encode_constant(FheType::EUint64, committed as u128)
            .expect("encode");
        self.ctx.harness().store().put(amt.pubkey().to_bytes(), digest, FheType::EUint64, None);
        self.ctx.enqueue_graph_execution(&mint_to_graph(), &[h.balance_ct, amt.pubkey()], &[h.balance_ct]);
        self.ctx.process_pending();
        self.ctx.register_ciphertext(&h.balance_ct);
    }

    fn balance(&mut self, h: &Holder) -> u128 {
        self.ctx.decrypt_from_store(&h.balance_ct)
    }

    fn transfer_ix(&self, from: &Pubkey, to: &Pubkey, from_ct: &Pubkey, to_ct: &Pubkey, amt_ct: &Pubkey, owner: &Pubkey) -> Instruction {
        let mut accts = vec![
            AccountMeta::new_readonly(*from, false),
            AccountMeta::new_readonly(*to, false),
            AccountMeta::new(*from_ct, false),
            AccountMeta::new(*to_ct, false),
            AccountMeta::new(*amt_ct, false),
        ];
        accts.extend(self.tail(owner));
        self.ix(vec![3, self.cpi_bump], accts)
    }

    fn view_ix(&self, ta: &Pubkey, bal: &Pubkey, view: &Pubkey, viewer: &Pubkey, owner: &Pubkey) -> Instruction {
        let mut accts = vec![
            AccountMeta::new_readonly(*ta, false),
            AccountMeta::new_readonly(*bal, false),
            AccountMeta::new(*view, true),
            AccountMeta::new_readonly(*viewer, false),
        ];
        accts.extend(self.tail(owner));
        self.ix(vec![40, self.cpi_bump], accts)
    }

    fn receipt_pda(&self, burned: &Pubkey) -> (Pubkey, u8) {
        Pubkey::find_program_address(&[b"pc_receipt", burned.as_ref()], &self.program_id)
    }

    fn unwrap_burn_ix(&self, vault: &Pubkey, h: &Holder, amt_ct: &Pubkey, burned: &Pubkey, burned_signer: bool, amount: u64) -> Instruction {
        let (receipt, rb) = self.receipt_pda(burned);
        let mut data = vec![31, rb, self.cpi_bump];
        data.extend_from_slice(&amount.to_le_bytes());
        let mut accts = vec![
            AccountMeta::new_readonly(*vault, false),
            AccountMeta::new(h.pda, false),
            AccountMeta::new(receipt, false),
            AccountMeta::new(h.balance_ct, false),
            AccountMeta::new(*amt_ct, false),
            AccountMeta::new(*burned, burned_signer),
        ];
        accts.extend(self.tail(&h.owner.pubkey()));
        self.ix(data, accts)
    }

    fn unwrap_decrypt_ix(&self, receipt: &Pubkey, req: &Pubkey, burned: &Pubkey, owner: &Pubkey) -> Instruction {
        let mut accts = vec![
            AccountMeta::new(*receipt, false),
            AccountMeta::new(*req, true),
            AccountMeta::new_readonly(*burned, false),
        ];
        let mut tail = self.tail(owner);
        tail[1] = AccountMeta::new_readonly(*self.ctx.config_pda(), false);
        accts.extend(tail);
        self.ix(vec![32, self.cpi_bump], accts)
    }

    fn unwrap_complete_ix(&self, receipt: &Pubkey, vault: &Pubkey, mint: &Pubkey, req: &Pubkey, vault_ata: &Pubkey, user_ata: &Pubkey, owner: &Pubkey) -> Instruction {
        self.ix(
            vec![33],
            vec![
                AccountMeta::new(*receipt, false),
                AccountMeta::new_readonly(*vault, false),
                AccountMeta::new_readonly(*mint, false),
                AccountMeta::new_readonly(*req, false),
                AccountMeta::new(*vault_ata, false),
                AccountMeta::new(*user_ata, false),
                AccountMeta::new_readonly(*owner, true),
                AccountMeta::new(*owner, false),
                AccountMeta::new_readonly(spl_token_id(), false),
            ],
        )
    }

    /// UnwrapBurn + settle the burn graph. Returns (receipt, burned_ct).
    fn burn(&mut self, side: &Side, h: &Holder, amount: u64) -> (Pubkey, Pubkey) {
        let amt_ct = self.ctx.create_input::<Uint64>(amount as u128, &self.program_id);
        let burned = Keypair::new();
        let ix = self.unwrap_burn_ix(&side.vault, h, &amt_ct, &burned.pubkey(), true, amount);
        self.send(&[ix], &[&h.owner, &burned]);
        self.ctx.register_ciphertext(&burned.pubkey());
        self.ctx.enqueue_graph_execution(&unwrap_burn_graph(), &[h.balance_ct, amt_ct], &[h.balance_ct, burned.pubkey()]);
        self.ctx.process_pending();
        self.ctx.register_ciphertext(&h.balance_ct);
        self.ctx.register_ciphertext(&burned.pubkey());
        (self.receipt_pda(&burned.pubkey()).0, burned.pubkey())
    }

    /// UnwrapDecrypt + the executor's decryption response. Returns the request.
    fn decrypt(&mut self, receipt: &Pubkey, burned: &Pubkey, h: &Holder) -> Pubkey {
        let req = Keypair::new();
        let ix = self.unwrap_decrypt_ix(receipt, &req.pubkey(), burned, &h.owner.pubkey());
        self.send(&[ix], &[&h.owner, &req]);
        let value = self.ctx.decrypt_from_store(burned) as u64;
        self.ctx
            .harness_mut()
            .tx_builder_mut()
            .respond_decryption(&req.pubkey().to_bytes(), &value.to_le_bytes())
            .expect("respond_decryption");
        req.pubkey()
    }
}

fn assert_custom(res: Result<(), String>, code: u32) {
    let e = res.expect_err("transaction should have failed");
    assert!(e.contains(&format!("Custom({code})")), "expected Custom({code:#x}), got: {e}");
}

fn assert_err_contains(res: Result<(), String>, needle: &str) {
    let e = res.expect_err("transaction should have failed");
    assert!(e.contains(needle), "expected {needle}, got: {e}");
}

// ── Wrap (Salmon finding 1) ──

#[test]
fn wrap_credits_exactly_the_spl_deposit() {
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    env.wrap(&side, &alice, 1);
    assert_eq!(env.balance(&alice), 1, "credit must equal the deposit");
    assert_eq!(env.spl_amount(&alice.ata), 999);
    assert_eq!(env.spl_amount(&side.vault_ata), 1);
}

#[test]
fn wrap_rejects_a_client_supplied_amount_ciphertext() {
    // Pre-epoch-2 exploit: wrap 1 SPL unit, pass an input ciphertext holding
    // 1_000_000, get credited 1_000_000.
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    let inflated = env.ctx.create_input::<Uint64>(1_000_000, &env.program_id);
    let ix = env.wrap_ix(&side.vault, &side.vault_ata, &alice, &inflated, false, 1);
    let res = env.try_send(&[ix], &[&alice.owner]);
    assert_custom(res, NOT_FRESH);
    assert_eq!(env.spl_amount(&alice.ata), 1_000, "no SPL moved");
}

#[test]
fn wrap_rejects_the_vault_of_another_mint() {
    let mut env = Env::new();
    let usdc = env.side();
    let doge = env.side();
    let alice = env.holder(&usdc, 0);
    // Alice holds DOGE SPL and tries to wrap it into her USDC pc-account.
    let doge_ata = env.spl_account(&doge.spl_mint, &alice.owner.pubkey(), 1_000);
    let amt = Keypair::new();
    let mut ix = env.wrap_ix(&doge.vault, &doge.vault_ata, &alice, &amt.pubkey(), true, 500);
    ix.accounts[2] = AccountMeta::new(doge_ata, false);
    let res = env.try_send(&[ix], &[&alice.owner, &amt]);
    assert_custom(res, BAD_VAULT);
}

#[test]
fn wrap_rejects_a_foreign_encrypt_program() {
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    let amt = Keypair::new();
    let mut ix = env.wrap_ix(&side.vault, &side.vault_ata, &alice, &amt.pubkey(), true, 1);
    ix.accounts[6] = AccountMeta::new_readonly(Pubkey::new_unique(), false);
    let res = env.try_send(&[ix], &[&alice.owner, &amt]);
    assert_custom(res, FOREIGN_ENCRYPT);
}

// ── Forged accounts (Salmon finding 2) ──

/// A look-alike TokenAccount owned by another program, claiming the victim's
/// balance ciphertext and the attacker as owner.
fn forge_token_account(env: &mut Env, victim: &Holder, attacker: &Pubkey) -> Pubkey {
    let mut data = env.ctx.get_account_data(&victim.pda).unwrap();
    data[32..64].copy_from_slice(attacker.as_ref());
    let key = Pubkey::new_unique();
    let lamports = env.ctx.minimum_balance(data.len());
    env.ctx.set_account(&key, data, &Pubkey::new_unique(), lamports);
    key
}

#[test]
fn forged_token_account_cannot_move_a_victims_balance() {
    let mut env = Env::new();
    let side = env.side();
    let victim = env.holder(&side, 1_000);
    env.wrap(&side, &victim, 1_000);
    let attacker = env.holder(&side, 0);
    let forged = forge_token_account(&mut env, &victim, &attacker.owner.pubkey());
    let amt = env.ctx.create_input::<Uint64>(1_000, &env.program_id);
    let ix = env.transfer_ix(&forged, &attacker.pda, &victim.balance_ct, &attacker.balance_ct, &amt, &attacker.owner.pubkey());
    let res = env.try_send(&[ix], &[&attacker.owner]);
    assert_err_contains(res, "IncorrectProgramId");
    assert_eq!(env.balance(&victim), 1_000);
}

#[test]
fn forged_token_account_cannot_copy_a_victims_balance() {
    let mut env = Env::new();
    let side = env.side();
    let victim = env.holder(&side, 1_000);
    let attacker = env.ctx.new_funded_keypair();
    let forged = forge_token_account(&mut env, &victim, &attacker.pubkey());
    let view = Keypair::new();
    let ix = env.view_ix(&forged, &victim.balance_ct, &view.pubkey(), &attacker.pubkey(), &attacker.pubkey());
    let res = env.try_send(&[ix], &[&attacker, &view]);
    assert_err_contains(res, "IncorrectProgramId");
}

#[test]
fn view_balance_still_works_for_the_owner() {
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    env.wrap(&side, &alice, 250);
    let view = Keypair::new();
    let viewer = Keypair::new();
    let ix = env.view_ix(&alice.pda, &alice.balance_ct, &view.pubkey(), &viewer.pubkey(), &alice.owner.pubkey());
    env.send(&[ix], &[&alice.owner, &view]);
    let data = env.ctx.get_account_data(&view.pubkey()).expect("view ciphertext created");
    assert_eq!(&data[34..66], viewer.pubkey().as_ref(), "copy authorized to the viewer key");
}

#[test]
fn self_transfer_is_rejected() {
    // Both graph outputs would land on one ciphertext and the credit would win.
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    env.wrap(&side, &alice, 100);
    let amt = env.ctx.create_input::<Uint64>(100, &env.program_id);
    let ix = env.transfer_ix(&alice.pda, &alice.pda, &alice.balance_ct, &alice.balance_ct, &amt, &alice.owner.pubkey());
    let res = env.try_send(&[ix], &[&alice.owner]);
    assert_custom(res, ALIASED);
}

#[test]
fn initialize_vault_requires_the_mint_authority() {
    let mut env = Env::new();
    let side = env.side();
    // A second vault for a fresh mint, attempted by a stranger.
    let authority = env.ctx.new_funded_keypair();
    let (mint, mint_bump) = Pubkey::find_program_address(&[b"pc_mint", authority.pubkey().as_ref()], &env.program_id);
    let mut data = vec![0u8, mint_bump, 6];
    data.extend_from_slice(authority.pubkey().as_ref());
    data.push(0);
    let payer = env.ctx.payer().pubkey();
    let ix = env.ix(data, vec![
        AccountMeta::new(mint, false),
        AccountMeta::new_readonly(authority.pubkey(), true),
        AccountMeta::new(payer, true),
        AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
    ]);
    env.send(&[ix], &[&authority]);
    let stranger = env.ctx.new_funded_keypair();
    let (vault, vb) = Pubkey::find_program_address(&[b"pc_vault", mint.as_ref()], &env.program_id);
    let ix = env.init_vault_ix(&vault, vb, &mint, &side.spl_mint, &stranger.pubkey());
    let res = env.try_send(&[ix], &[&stranger]);
    assert_custom(res, NOT_MINT_AUTHORITY);
}

// ── Unwrap (Salmon finding 3) ──

#[test]
fn unwrap_end_to_end_pays_out_the_burn() {
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    env.wrap(&side, &alice, 600);
    let (receipt, burned) = env.burn(&side, &alice, 400);
    assert_eq!(env.balance(&alice), 200);
    let req = env.decrypt(&receipt, &burned, &alice);
    let ix = env.unwrap_complete_ix(&receipt, &side.vault, &side.mint, &req, &side.vault_ata, &alice.ata, &alice.owner.pubkey());
    env.send(&[ix], &[&alice.owner]);
    assert_eq!(env.spl_amount(&alice.ata), 800, "400 + 400 back");
    assert_eq!(env.spl_amount(&side.vault_ata), 200);
    let gone = env.ctx.get_account_data(&receipt).map(|d| d.is_empty() || d.iter().all(|b| *b == 0)).unwrap_or(true);
    assert!(gone, "receipt closed");
}

#[test]
fn unwrap_burn_rejects_an_existing_ciphertext_as_output() {
    // Naming someone's balance ciphertext as the burn output would overwrite it.
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    let victim = env.holder(&side, 0);
    env.wrap(&side, &alice, 100);
    let amt = env.ctx.create_input::<Uint64>(100, &env.program_id);
    let ix = env.unwrap_burn_ix(&side.vault, &alice, &amt, &victim.balance_ct, false, 100);
    let res = env.try_send(&[ix], &[&alice.owner]);
    assert_custom(res, NOT_FRESH);
}

#[test]
fn unwrap_decrypt_rejects_a_substituted_ciphertext() {
    // Pre-epoch-2 exploit: decrypt an input ciphertext holding 1_000_000
    // instead of the burn, and complete the unwrap against it.
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    env.wrap(&side, &alice, 10);
    let (receipt, _burned) = env.burn(&side, &alice, 1_000_000);
    let fake = env.ctx.create_input::<Uint64>(1_000_000, &env.program_id);
    let req = Keypair::new();
    let ix = env.unwrap_decrypt_ix(&receipt, &req.pubkey(), &fake, &alice.owner.pubkey());
    let res = env.try_send(&[ix], &[&alice.owner, &req]);
    assert_custom(res, RECEIPT_MISMATCH);
}

#[test]
fn unwrap_decrypt_runs_once() {
    let mut env = Env::new();
    let side = env.side();
    let alice = env.holder(&side, 1_000);
    env.wrap(&side, &alice, 100);
    let (receipt, burned) = env.burn(&side, &alice, 100);
    env.decrypt(&receipt, &burned, &alice);
    let req2 = Keypair::new();
    let ix = env.unwrap_decrypt_ix(&receipt, &req2.pubkey(), &burned, &alice.owner.pubkey());
    let res = env.try_send(&[ix], &[&alice.owner, &req2]);
    assert_custom(res, DECRYPT_ALREADY_REQUESTED);
}

#[test]
fn unwrap_complete_rejects_another_mints_vault() {
    // Pre-epoch-2: the receipt stored no mint, so a burn of one pc mint could
    // be completed against the other mint's vault.
    let mut env = Env::new();
    let usdc = env.side();
    let doge = env.side();
    let alice = env.holder(&doge, 1_000);
    env.wrap(&doge, &alice, 500);
    // Fund the USDC vault so a cross-mint payout would have something to take.
    let usdc_whale = env.holder(&usdc, 1_000);
    env.wrap(&usdc, &usdc_whale, 1_000);
    let (receipt, burned) = env.burn(&doge, &alice, 500);
    let req = env.decrypt(&receipt, &burned, &alice);
    let alice_usdc_ata = env.spl_account(&usdc.spl_mint, &alice.owner.pubkey(), 0);
    let ix = env.unwrap_complete_ix(&receipt, &usdc.vault, &usdc.mint, &req, &usdc.vault_ata, &alice_usdc_ata, &alice.owner.pubkey());
    let res = env.try_send(&[ix], &[&alice.owner]);
    assert_custom(res, RECEIPT_MISMATCH);
    assert_eq!(env.spl_amount(&usdc.vault_ata), 1_000);
    // The honest completion still works afterwards.
    let ix = env.unwrap_complete_ix(&receipt, &doge.vault, &doge.mint, &req, &doge.vault_ata, &alice.ata, &alice.owner.pubkey());
    env.send(&[ix], &[&alice.owner]);
    assert_eq!(env.spl_amount(&alice.ata), 1_000);
    let _ = (usdc.vault_bump, doge.authority.pubkey());
}
