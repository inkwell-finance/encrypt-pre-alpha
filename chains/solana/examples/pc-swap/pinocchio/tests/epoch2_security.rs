// Copyright (c) 2026 Inkwell Finance
// SPDX-License-Identifier: BSD-3-Clause-Clear

//! Epoch-2 regression tests for pc-swap: a forged pool, a fake "pc-token"
//! program and a forged LP position are refused before any CPI; the honest
//! CreatePool → CreateLpPosition → Swap path still goes through the pinned
//! pc-token program.
//!
//! Runs against the `localnet` builds in `target/deploy-localnet/`, deployed
//! at the epoch-2 program ids, with the Encrypt program binary in `bin/`.

use std::str::FromStr;

use encrypt_dev::runtime::SolanaRuntime;
use encrypt_solana_test::litesvm::EncryptTestContext;
use encrypt_types::encrypted::Uint64;
use solana_sdk::instruction::{AccountMeta, Instruction};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::transaction::Transaction;

const PC_SWAP_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../../target/deploy-localnet/pc_swap.so");
const PC_TOKEN_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../../target/deploy-localnet/pc_token.so");

const SYSTEM_PROGRAM: Pubkey = Pubkey::new_from_array([0u8; 32]);

const FOREIGN_PC_TOKEN: u32 = 0x202;
const BAD_PC_TOKEN_CPI_AUTH: u32 = 0x203;
const BAD_POOL: u32 = 0x204;
const WRONG_CALLER: u32 = 0x201;

fn pc_swap_id() -> Pubkey {
    Pubkey::from_str("BqJMtWQYiis1264KqBN3wssNqTofK5RuP67hkZYvb4SD").unwrap()
}
fn pc_token_id() -> Pubkey {
    Pubkey::from_str("3kjJEx6G1XdrtixFg9f8eGRAyzW5QmhsJQq6qCUZYhXZ").unwrap()
}

struct Env {
    ctx: EncryptTestContext,
    swap: Pubkey,
    token: Pubkey,
    swap_auth: Pubkey,
    swap_bump: u8,
    token_auth: Pubkey,
    token_bump: u8,
}

struct Pool {
    pool: Pubkey,
    mint_a: Pubkey,
    mint_b: Pubkey,
    vault_a: Pubkey,
    vault_b: Pubkey,
    vault_a_ct: Pubkey,
    vault_b_ct: Pubkey,
    reserve_a: Pubkey,
    reserve_b: Pubkey,
}

impl Env {
    fn new() -> Self {
        let mut ctx = EncryptTestContext::new_default();
        let swap = pc_swap_id();
        let token = pc_token_id();
        ctx.deploy_program_at(&token, PC_TOKEN_PATH);
        ctx.deploy_program_at(&swap, PC_SWAP_PATH);
        let (swap_auth, swap_bump) = ctx.cpi_authority_for(&swap);
        let (token_auth, token_bump) = ctx.cpi_authority_for(&token);
        Env { ctx, swap, token, swap_auth, swap_bump, token_auth, token_bump }
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

    fn send(&mut self, ixs: &[Instruction], signers: &[&Keypair]) {
        self.try_send(ixs, signers).expect("transaction failed");
    }

    fn pc_mint(&mut self) -> Pubkey {
        let authority = self.ctx.new_funded_keypair();
        let (mint, bump) = Pubkey::find_program_address(&[b"pc_mint", authority.pubkey().as_ref()], &self.token);
        let mut data = vec![0u8, bump, 6];
        data.extend_from_slice(authority.pubkey().as_ref());
        data.push(0);
        let payer = self.ctx.payer().pubkey();
        let ix = Instruction::new_with_bytes(self.token, &data, vec![
            AccountMeta::new(mint, false),
            AccountMeta::new_readonly(authority.pubkey(), true),
            AccountMeta::new(payer, true),
            AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
        ]);
        self.send(&[ix], &[&authority]);
        mint
    }

    fn token_tail(&self, payer: &Pubkey) -> Vec<AccountMeta> {
        vec![
            AccountMeta::new_readonly(*self.ctx.program_id(), false),
            AccountMeta::new(*self.ctx.config_pda(), false),
            AccountMeta::new(*self.ctx.deposit_pda(), false),
            AccountMeta::new_readonly(self.token_auth, false),
            AccountMeta::new_readonly(self.token, false),
            AccountMeta::new_readonly(*self.ctx.network_encryption_key_pda(), false),
            AccountMeta::new(*payer, true),
            AccountMeta::new_readonly(*self.ctx.event_authority(), false),
            AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
        ]
    }

    /// pc-token account for `owner`; returns (pda, balance_ct).
    fn pc_account(&mut self, mint: &Pubkey, owner: &Pubkey) -> (Pubkey, Pubkey) {
        let (pda, bump) = Pubkey::find_program_address(&[b"pc_account", mint.as_ref(), owner.as_ref()], &self.token);
        let bal = Keypair::new();
        let payer = self.ctx.payer().pubkey();
        let mut accts = vec![
            AccountMeta::new(pda, false),
            AccountMeta::new_readonly(*mint, false),
            AccountMeta::new_readonly(*owner, false),
            AccountMeta::new(bal.pubkey(), true),
        ];
        accts.extend(self.token_tail(&payer));
        let ix = Instruction::new_with_bytes(self.token, &[1, bump, self.token_bump], accts);
        self.send(&[ix], &[&bal]);
        self.ctx.register_ciphertext(&bal.pubkey());
        (pda, bal.pubkey())
    }

    /// The swap-side Encrypt tail: ep, cfg, dep, swap cpi auth, swap, token cpi auth, token.
    fn swap_programs(&self) -> Vec<AccountMeta> {
        vec![
            AccountMeta::new_readonly(*self.ctx.program_id(), false),
            AccountMeta::new(*self.ctx.config_pda(), false),
            AccountMeta::new(*self.ctx.deposit_pda(), false),
            AccountMeta::new_readonly(self.swap_auth, false),
            AccountMeta::new_readonly(self.swap, false),
            AccountMeta::new_readonly(self.token_auth, false),
            AccountMeta::new_readonly(self.token, false),
        ]
    }

    fn swap_end(&self, payer: &Pubkey) -> Vec<AccountMeta> {
        vec![
            AccountMeta::new_readonly(*self.ctx.network_encryption_key_pda(), false),
            AccountMeta::new(*payer, true),
            AccountMeta::new_readonly(*self.ctx.event_authority(), false),
            AccountMeta::new_readonly(SYSTEM_PROGRAM, false),
        ]
    }

    fn create_pool(&mut self) -> Pool {
        let mint_a = self.pc_mint();
        let mint_b = self.pc_mint();
        let (pool, pool_bump) = Pubkey::find_program_address(&[b"pc_pool", mint_a.as_ref(), mint_b.as_ref()], &self.swap);
        let (vault_a, va_bump) = Pubkey::find_program_address(&[b"pc_account", mint_a.as_ref(), pool.as_ref()], &self.token);
        let (vault_b, vb_bump) = Pubkey::find_program_address(&[b"pc_account", mint_b.as_ref(), pool.as_ref()], &self.token);
        let (va_ct, vb_ct, ra, rb, ts) = (Keypair::new(), Keypair::new(), Keypair::new(), Keypair::new(), Keypair::new());
        let payer = self.ctx.payer().pubkey();
        let mut accts = vec![
            AccountMeta::new(pool, false),
            AccountMeta::new_readonly(mint_a, false),
            AccountMeta::new_readonly(mint_b, false),
            AccountMeta::new(vault_a, false),
            AccountMeta::new(vault_b, false),
            AccountMeta::new(va_ct.pubkey(), true),
            AccountMeta::new(vb_ct.pubkey(), true),
            AccountMeta::new(ra.pubkey(), true),
            AccountMeta::new(rb.pubkey(), true),
            AccountMeta::new(ts.pubkey(), true),
        ];
        accts.extend(self.swap_programs());
        accts.extend(self.swap_end(&payer));
        let ix = Instruction::new_with_bytes(self.swap, &[0, pool_bump, va_bump, vb_bump, self.swap_bump, self.token_bump, 0], accts);
        self.send(&[ix], &[&va_ct, &vb_ct, &ra, &rb, &ts]);
        for k in [&va_ct, &vb_ct, &ra, &rb, &ts] {
            self.ctx.register_ciphertext(&k.pubkey());
        }
        Pool {
            pool, mint_a, mint_b, vault_a, vault_b,
            vault_a_ct: va_ct.pubkey(), vault_b_ct: vb_ct.pubkey(),
            reserve_a: ra.pubkey(), reserve_b: rb.pubkey(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn swap_ix(
        &self, pool: &Pool, user: &Pubkey,
        user_in: (Pubkey, Pubkey), user_out: (Pubkey, Pubkey),
        amt_in: &Pubkey, min_out: &Pubkey, amt_out: &Pubkey, receipt: &Pubkey,
    ) -> Instruction {
        let mut accts = vec![
            AccountMeta::new(pool.pool, false),
            AccountMeta::new(user_in.0, false),
            AccountMeta::new(user_out.0, false),
            AccountMeta::new(pool.vault_a, false),
            AccountMeta::new(pool.vault_b, false),
            AccountMeta::new(user_in.1, false),
            AccountMeta::new(user_out.1, false),
            AccountMeta::new(pool.vault_a_ct, false),
            AccountMeta::new(pool.vault_b_ct, false),
            AccountMeta::new(pool.reserve_a, false),
            AccountMeta::new(pool.reserve_b, false),
            AccountMeta::new(*amt_in, false),
            AccountMeta::new(*min_out, false),
            AccountMeta::new(*amt_out, false),
            AccountMeta::new(*receipt, true),
        ];
        accts.extend(self.swap_programs());
        accts.extend(self.swap_end(user));
        Instruction::new_with_bytes(self.swap, &[1, self.swap_bump, self.token_bump, 0], accts)
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

/// A swap instruction against `pool` whose user side is throwaway keys. Every
/// check under test fails before any of these accounts is read.
fn dummy_swap(env: &mut Env, pool: &Pool) -> (Instruction, Keypair, Keypair) {
    let user = env.ctx.new_funded_keypair();
    let receipt = Keypair::new();
    let k = || Pubkey::new_unique();
    let ix = env.swap_ix(pool, &user.pubkey(), (k(), k()), (k(), k()), &k(), &k(), &k(), &receipt.pubkey());
    (ix, user, receipt)
}

// ── Salmon finding 4 ──

#[test]
fn swap_rejects_a_fake_pc_token_program() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let (mut ix, user, receipt) = dummy_swap(&mut env, &pool);
    // A program the attacker controls in the pc_token_program slot (index 21).
    ix.accounts[21] = AccountMeta::new_readonly(Pubkey::new_unique(), false);
    let res = env.try_send(&[ix], &[&user, &receipt]);
    assert_custom(res, FOREIGN_PC_TOKEN);
}

#[test]
fn swap_rejects_a_fake_pc_token_cpi_authority() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let (mut ix, user, receipt) = dummy_swap(&mut env, &pool);
    ix.accounts[20] = AccountMeta::new_readonly(Pubkey::new_unique(), false);
    let res = env.try_send(&[ix], &[&user, &receipt]);
    assert_custom(res, BAD_PC_TOKEN_CPI_AUTH);
}

#[test]
fn swap_rejects_a_foreign_caller_program() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let (mut ix, user, receipt) = dummy_swap(&mut env, &pool);
    ix.accounts[19] = AccountMeta::new_readonly(Pubkey::new_unique(), false);
    let res = env.try_send(&[ix], &[&user, &receipt]);
    assert_custom(res, WRONG_CALLER);
}

#[test]
fn swap_rejects_a_forged_pool() {
    // A look-alike pool owned by another program, copying the real pool's
    // mints and bump (so its seeds would sign as the real pool PDA).
    let mut env = Env::new();
    let pool = env.create_pool();
    let data = env.ctx.get_account_data(&pool.pool).unwrap();
    let forged = Pubkey::new_unique();
    let lamports = env.ctx.minimum_balance(data.len());
    env.ctx.set_account(&forged, data, &Pubkey::new_unique(), lamports);
    let (mut ix, user, receipt) = dummy_swap(&mut env, &pool);
    ix.accounts[0] = AccountMeta::new(forged, false);
    let res = env.try_send(&[ix], &[&user, &receipt]);
    assert_err_contains(res, "IncorrectProgramId");
}

#[test]
fn swap_rejects_a_pool_account_that_is_not_its_pda() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let data = env.ctx.get_account_data(&pool.pool).unwrap();
    let elsewhere = Pubkey::new_unique();
    let lamports = env.ctx.minimum_balance(data.len());
    let swap = env.swap;
    env.ctx.set_account(&elsewhere, data, &swap, lamports);
    let (mut ix, user, receipt) = dummy_swap(&mut env, &pool);
    ix.accounts[0] = AccountMeta::new(elsewhere, false);
    let res = env.try_send(&[ix], &[&user, &receipt]);
    assert_custom(res, BAD_POOL);
}

#[test]
fn remove_liquidity_rejects_a_forged_lp_position() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let user = env.ctx.new_funded_keypair();
    // LpPosition layout: pool, owner, balance ct, bump — owned by another program.
    let mut lp = vec![0u8; 97];
    lp[0..32].copy_from_slice(pool.pool.as_ref());
    lp[32..64].copy_from_slice(user.pubkey().as_ref());
    let lp_ct = env.ctx.create_input::<Uint64>(1_000_000, &pc_swap_id());
    lp[64..96].copy_from_slice(lp_ct.as_ref());
    let forged = Pubkey::new_unique();
    let lamports = env.ctx.minimum_balance(97);
    env.ctx.set_account(&forged, lp, &Pubkey::new_unique(), lamports);

    let pool_data = env.ctx.get_account_data(&pool.pool).unwrap();
    let total_supply = Pubkey::new_from_array(pool_data[192..224].try_into().unwrap());
    let k = || Pubkey::new_unique();
    let mut accts = vec![
        AccountMeta::new(pool.pool, false),
        AccountMeta::new_readonly(forged, false),
        AccountMeta::new(k(), false),
        AccountMeta::new(k(), false),
        AccountMeta::new(pool.vault_a, false),
        AccountMeta::new(pool.vault_b, false),
        AccountMeta::new(k(), false),
        AccountMeta::new(k(), false),
        AccountMeta::new(pool.vault_a_ct, false),
        AccountMeta::new(pool.vault_b_ct, false),
        AccountMeta::new(pool.reserve_a, false),
        AccountMeta::new(pool.reserve_b, false),
        AccountMeta::new(total_supply, false),
        AccountMeta::new(k(), false),
        AccountMeta::new(lp_ct, false),
        AccountMeta::new(k(), false),
        AccountMeta::new(k(), false),
    ];
    accts.extend(env.swap_programs());
    accts.extend(env.swap_end(&user.pubkey()));
    let ix = Instruction::new_with_bytes(env.swap, &[4, env.swap_bump, env.token_bump], accts);
    let res = env.try_send(&[ix], &[&user]);
    assert_err_contains(res, "IncorrectProgramId");
}

// ── Honest paths through the pinned pc-token ──

#[test]
fn create_pool_and_lp_position_go_through_pinned_pc_token() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let data = env.ctx.get_account_data(&pool.pool).expect("pool created");
    assert_eq!(data.len(), 226);
    assert_eq!(&data[0..32], pool.mint_a.as_ref());
    assert_eq!(&data[32..64], pool.mint_b.as_ref());
    // The vaults are real pc-token accounts owned by the pool PDA.
    let va = env.ctx.get_account_data(&pool.vault_a).expect("vault a");
    assert_eq!(&va[32..64], pool.pool.as_ref());

    let user = env.ctx.new_funded_keypair();
    let (lp, lp_bump) = Pubkey::find_program_address(&[b"pc_lp", pool.pool.as_ref(), user.pubkey().as_ref()], &env.swap);
    let bal = Keypair::new();
    let payer = env.ctx.payer().pubkey();
    let mut accts = vec![
        AccountMeta::new(lp, false),
        AccountMeta::new_readonly(pool.pool, false),
        AccountMeta::new_readonly(user.pubkey(), false),
        AccountMeta::new(bal.pubkey(), true),
        AccountMeta::new_readonly(*env.ctx.program_id(), false),
        AccountMeta::new(*env.ctx.config_pda(), false),
        AccountMeta::new(*env.ctx.deposit_pda(), false),
        AccountMeta::new_readonly(env.swap_auth, false),
        AccountMeta::new_readonly(env.swap, false),
    ];
    accts.extend(env.swap_end(&payer));
    let ix = Instruction::new_with_bytes(env.swap, &[5, lp_bump, env.swap_bump], accts);
    env.send(&[ix], &[&bal]);
    assert_eq!(env.ctx.get_account_data(&lp).expect("lp created").len(), 97);
}

#[test]
fn honest_swap_goes_through_pinned_pc_token() {
    let mut env = Env::new();
    let pool = env.create_pool();
    let user = env.ctx.new_funded_keypair();
    let user_in = env.pc_account(&pool.mint_a, &user.pubkey());
    let user_out = env.pc_account(&pool.mint_b, &user.pubkey());
    let swap = env.swap;
    let amt_in = env.ctx.create_input::<Uint64>(10, &swap);
    let min_out = env.ctx.create_input::<Uint64>(0, &swap);
    let amt_out = env.ctx.create_input::<Uint64>(0, &swap);
    let receipt = Keypair::new();
    let ix = env.swap_ix(&pool, &user.pubkey(), user_in, user_out, &amt_in, &min_out, &amt_out, &receipt.pubkey());
    env.send(&[ix], &[&user, &receipt]);
    let _ = (pool.vault_b, pool.reserve_b);
}
