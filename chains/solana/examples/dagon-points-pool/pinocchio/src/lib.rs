// Copyright (c) Inkwell Finance.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#![allow(unexpected_cfgs)]

//! # dagon-points-pool
//!
//! Single-program FHE confidential AMM for the Dagon points farm. Owns BOTH the pool reserves
//! AND user balances under one CPI authority, so a single FHE graph can atomically debit a
//! user, update reserves, and credit the other side — no chained CPIs, no multi-op atomicity
//! assumptions about the off-chain executor.
//!
//! ## Why this exists (vs. PC-Token + PC-Swap)
//!
//! Encrypt's pre-alpha doesn't expose a primitive that lets a single FHE graph span
//! ciphertexts authorized to different programs. Composing PC-Token (user balances) with
//! PC-Swap (pool reserves) requires chained-CPI with `copy_ciphertext` to bridge auth, which
//! splits the work across multiple FHE ops. Without a documented "atomic op group" guarantee
//! from the executor, that design has a real failure window: pool moves while user-leg
//! never commits, leaving the pool desynchronized.
//!
//! This program collapses both sides into one auth namespace. One graph. One executor commit.
//! One atomic state change. Trades upstream-composability for safety.
//!
//! ## Instructions
//!
//! 0. `InitPool` — singleton pool + both SPL vaults
//! 1. `SeedLiquidity` — deployer adds plaintext liquidity to reserves
//! 2. `InitUserBalance` — per-user-per-side encrypted balance account
//! 3. `WrapDeposit` — SPL → vault → encrypted user balance (single FHE op: `balance += amount`)
//! 4. `SettleSwap` — the headline atomic op: 7 inputs, 5 outputs, one graph
//! 5. `ViewBalance` — `copy_ciphertext` for client-side decrypt on demand
//! 6. `AcceptDepositFromCold` — consume an incoming pc-token-cold ticket; create UserBalance
//!    referencing the now-pool-authorized CT (the source side rotated auth into us)
//! 7. `WithdrawToCold` — rotate authority of user's hot balance back to pc-token-cold's
//!    CPI authority, CPI into pc-token-cold to write a matching ticket, close UserBalance
//!
//! ## FHE type discipline
//!
//! Everything is `EUint128` so the constant-product math `(amount * 997) * reserve` doesn't
//! overflow. Same as PC-Swap's choice; intentionally compatible for any future composability.

use encrypt_dsl::prelude::encrypt_fn;
use encrypt_pinocchio::accounts;
use encrypt_pinocchio::EncryptContext;
use encrypt_types::encrypted::{EUint128, Uint128};
use pinocchio::{
    cpi::{invoke_signed, Seed, Signer},
    entrypoint,
    error::ProgramError,
    instruction::{InstructionAccount, InstructionView},
    AccountView, Address, ProgramResult,
};
use pinocchio_system::instructions::CreateAccount;
use pinocchio_token::instructions::Transfer as SplTransfer;

entrypoint!(process_instruction);

pub const ID: Address = Address::new_from_array([7u8; 32]);

/// Hardcoded trusted pc-token-cold program ID (Inkwell devnet:
/// `FaikTJS85AAPbpmxsJ9ijWFKmR7gJJNR6oppYXQdK6SJ`). Bound here for the cold↔hot transfer
/// flow: deposit_from_cold reads tickets created by pc-token-cold, withdraw_to_cold rotates
/// authority to pc-token-cold's CPI authority and CPIs into pc-token-cold to create a
/// matching ticket.
///
/// **If you redeploy pc-token-cold, this constant MUST be updated.**
pub const TRUSTED_COLD_PROGRAM_ID: Address = Address::new_from_array([
    0xd8, 0xa6, 0xfd, 0x58, 0xd2, 0xa5, 0x4c, 0xc4, 0xcd, 0x9c, 0x1b, 0xc1, 0xe8, 0x73, 0xa7, 0xaa,
    0xc5, 0xfe, 0xbe, 0x85, 0x1a, 0x7b, 0xb6, 0x38, 0x14, 0x01, 0x36, 0xc5, 0x0e, 0xed, 0xbb, 0xdf,
]);

/// pc-token-cold ticket layout (mirrored from `pc_token_cold::Ticket` for in-place reads).
/// PDA seeds in pc-token-cold: `[b"pc_ticket", balance_ct]`.
///   [0..32]   balance_ct
///   [32..64]  recipient
///   [64..96]  mint
///   [96]      bump
const TICKET_BALANCE_CT_OFFSET: usize = 0;
const TICKET_RECIPIENT_OFFSET: usize = 32;
const TICKET_MINT_OFFSET: usize = 64;
const TICKET_LEN: usize = 104; // 32 + 32 + 32 + 1 + 7 padding

/// Discriminator for pc-token-cold's `create_inbound_ticket_cpi` ix.
const PC_TOKEN_COLD_IX_CREATE_INBOUND_TICKET: u8 = 7;

/// Discriminator for pc-token-cold's `release_spl` ix (settlement-program escape hatch
/// invoked from `unwrap_complete` to drain SPL backing from cold's per-mint vault on
/// behalf of an FHE-burn this program performed against one of its hot UserBalances).
const PC_TOKEN_COLD_IX_RELEASE_SPL: u8 = 12;

// ── Sides ──
//
// Two-sided pool. SIDE_A = "USDC" (the unit-of-account token). SIDE_B = "SOL" (the volatile
// token). Direction byte in SettleSwap: 0 = A→B, 1 = B→A.
pub const SIDE_A: u8 = 0;
pub const SIDE_B: u8 = 1;

// ── Account layouts ──

/// Pool — singleton PDA at seeds `[b"dagon_pool"]`.
/// Holds the encrypted reserves, total-supply ciphertext, public price ciphertext, and the
/// SPL mint + vault addresses for both sides.
#[repr(C)]
pub struct Pool {
    pub mint_a: [u8; 32],          // SPL mint for side A (mock-USDC)
    pub mint_b: [u8; 32],          // SPL mint for side B (WSOL)
    pub vault_a: [u8; 32],         // ATA holding SPL backing of side A reserves
    pub vault_b: [u8; 32],         // ATA holding SPL backing of side B reserves
    pub reserve_a: [u8; 32],       // EUint128 ciphertext of side A reserves
    pub reserve_b: [u8; 32],       // EUint128 ciphertext of side B reserves
    pub price_ct: [u8; 32],        // PUBLIC EUint128 ciphertext: B per A * 1_000_000
    pub admin: [u8; 32],           // Authority for SeedLiquidity (one-time seed at init)
    pub is_initialized: u8,
    pub bump: u8,
    pub _pad: [u8; 6],             // explicit padding for repr(C) alignment / future use
}

impl Pool {
    pub const LEN: usize = core::mem::size_of::<Self>();
    pub fn from_bytes(d: &[u8]) -> Result<&Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &*(d.as_ptr() as *const Self) })
    }
    pub fn from_bytes_mut(d: &mut [u8]) -> Result<&mut Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &mut *(d.as_mut_ptr() as *mut Self) })
    }
}

/// UserBalance — per-user, per-side encrypted balance.
/// PDA seeds: `[b"dagon_user", owner.as_ref(), [side]]`.
#[repr(C)]
pub struct UserBalance {
    pub owner: [u8; 32],
    pub balance_ct: [u8; 32],
    pub side: u8,
    pub bump: u8,
    pub _pad: [u8; 6],
}

impl UserBalance {
    pub const LEN: usize = core::mem::size_of::<Self>();
    pub fn from_bytes(d: &[u8]) -> Result<&Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &*(d.as_ptr() as *const Self) })
    }
    pub fn from_bytes_mut(d: &mut [u8]) -> Result<&mut Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &mut *(d.as_mut_ptr() as *mut Self) })
    }
}

/// Temporary receipt for hot → SPL unwrap. Created by `unwrap_burn`, populated with the
/// digest by `unwrap_decrypt`, consumed + closed by `unwrap_complete`.
/// PDA seeds: `[b"pool_receipt", burned_ct]` — distinct from cold's `[b"pc_receipt", ...]`
/// to avoid PDA collision when both programs are mid-unwrap on the same user.
#[repr(C)]
pub struct WithdrawalReceipt {
    pub owner: [u8; 32],
    pub amount: [u8; 8],           // requested plaintext amount (u64, SPL-bound)
    pub pending_digest: [u8; 32],  // digest snapshot taken at request_decryption
    pub mint: [u8; 32],            // SPL mint to release on complete
    pub side: u8,                  // SIDE_A or SIDE_B — selects which UB was burned
    pub bump: u8,
    pub _pad: [u8; 6],
}

impl WithdrawalReceipt {
    pub const LEN: usize = core::mem::size_of::<Self>();
    pub fn from_bytes(d: &[u8]) -> Result<&Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &*(d.as_ptr() as *const Self) })
    }
    pub fn from_bytes_mut(d: &mut [u8]) -> Result<&mut Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &mut *(d.as_mut_ptr() as *mut Self) })
    }
    pub fn requested_amount(&self) -> u64 { u64::from_le_bytes(self.amount) }
}

fn minimum_balance(s: usize) -> u64 { (s as u64 + 128) * 6960 }

// ── FHE Graphs ──

/// Mint-to: `balance += amount`. Used by Wrap (SPL → encrypted balance).
#[encrypt_fn]
fn mint_to_graph(balance: EUint128, amount: EUint128) -> EUint128 {
    balance + amount
}

/// Add to plaintext reserve: `reserve += amount`. Used by SeedLiquidity. Same op as
/// `mint_to_graph` but kept distinct for grep-ability of intent.
#[encrypt_fn]
fn seed_reserve_graph(reserve: EUint128, amount: EUint128) -> EUint128 {
    reserve + amount
}

/// Conditional burn for hot → SPL unwrap. `burned = amount if balance >= amount else 0`.
/// The decrypted burned value is used by `unwrap_complete` to gate the cold.release_spl CPI.
/// Mirrors pc-token-cold's `unwrap_burn_graph`.
#[encrypt_fn]
fn unwrap_burn_graph(balance: EUint128, amount: EUint128) -> (EUint128, EUint128) {
    let s = balance >= amount;
    let new_balance = if s { balance - amount } else { balance };
    let burned = if s { amount } else { amount - amount };
    (new_balance, burned)
}

/// Drain-all burn for `close_balance`. Always sets balance to 0 and outputs the pre-burn
/// balance as `burned`. close_complete uses the decrypted `burned` to drive cold.release_spl
/// for the corresponding SPL release, then closes the UB + balance_ct + burned_ct + receipt
/// + decryption request.
#[encrypt_fn]
fn close_burn_graph(balance: EUint128) -> (EUint128, EUint128) {
    let zero = balance - balance;
    (zero, balance)
}

/// Atomic swap graph — the headline. Seven inputs, five outputs, one FHE op.
///
/// Reads:  reserve_in, reserve_out, user_in, user_out, amount_in, min_out, current_price
/// Writes: new_reserve_in, new_reserve_out, new_user_in, new_user_out, new_price
///
/// Validity = `k_invariant_ok && slippage_ok && user_has_balance`. All three checks gate
/// the same writes. If any condition is false the entire op no-ops — pool unchanged, user
/// unchanged, price unchanged. There is no asymmetric-failure window because every output
/// is gated on the unified `valid` flag.
#[encrypt_fn]
fn atomic_swap_graph(
    reserve_in: EUint128,
    reserve_out: EUint128,
    user_in: EUint128,
    user_out: EUint128,
    amount_in: EUint128,
    min_out: EUint128,
    current_price: EUint128,
) -> (EUint128, EUint128, EUint128, EUint128, EUint128) {
    // Constant-product with 0.3% fee.
    let amount_in_with_fee = amount_in * 997;
    let numerator = amount_in_with_fee * reserve_out;
    let denominator = (reserve_in * 1000) + amount_in_with_fee;
    let amount_out = numerator / denominator;
    let new_reserve_in = reserve_in + amount_in;
    let new_reserve_out = reserve_out - amount_out;

    // Three independent guards.
    let old_k = reserve_in * reserve_out;
    let new_k = new_reserve_in * new_reserve_out;
    let k_ok = new_k >= old_k;
    let slippage_ok = amount_out >= min_out;
    let user_has_balance = user_in >= amount_in;

    // valid = k_ok && slippage_ok && user_has_balance (FHE-AND via nested if; same trick as
    // upstream PC-Swap's valid-flag pattern).
    let valid_ks = if k_ok { slippage_ok } else { k_ok };
    let valid = if valid_ks { user_has_balance } else { valid_ks };

    let final_reserve_in = if valid { new_reserve_in } else { reserve_in };
    let final_reserve_out = if valid { new_reserve_out } else { reserve_out };
    let final_user_in = if valid { user_in - amount_in } else { user_in };
    let final_user_out = if valid { user_out + amount_out } else { user_out };

    let new_price = (final_reserve_out * 1_000_000) / (final_reserve_in + 1);
    let final_price = if valid { new_price } else { current_price };

    (final_reserve_in, final_reserve_out, final_user_in, final_user_out, final_price)
}

// ── Dispatch ──

fn process_instruction(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    match data.split_first() {
        Some((&0, rest)) => init_pool(program_id, accounts, rest),
        Some((&1, rest)) => seed_liquidity(accounts, rest),
        Some((&2, rest)) => init_user_balance(program_id, accounts, rest),
        Some((&3, rest)) => wrap_deposit(accounts, rest),
        Some((&4, rest)) => settle_swap(accounts, rest),
        Some((&5, rest)) => view_balance(accounts, rest),
        Some((&6, rest)) => accept_deposit_from_cold(program_id, accounts, rest),
        Some((&7, rest)) => withdraw_to_cold(accounts, rest),
        Some((&8, rest)) => merge_deposit_into_existing(accounts, rest),
        // 9-11: hot → SPL unwrap. Same multi-step pattern as cold's: burn (FHE),
        // decrypt (executor), complete (verify + CPI to cold.release_spl).
        Some((&9, rest)) => unwrap_burn(program_id, accounts, rest),
        Some((&10, rest)) => unwrap_decrypt(accounts, rest),
        Some((&11, rest)) => unwrap_complete(accounts, rest),
        // 12: admin-only one-time migration of legacy pool.vault_x SPL into cold.M.vault_ata.
        // After running for both sides, all SPL backing lives in cold and pool.vault_x is empty.
        Some((&12, rest)) => migrate_vault_to_cold(accounts, rest),
        // 13-14: drain-all + close. Reuses unwrap_decrypt (ix=10) for the middle step.
        Some((&13, rest)) => close_burn(program_id, accounts, rest),
        Some((&14, rest)) => close_complete(accounts, rest),
        _ => Err(ProgramError::InvalidInstructionData),
    }
}

// ── 0: InitPool ──
//
// Account layout:
//   [0]  pool_acct (PDA, will be created)
//   [1]  mint_a (SPL mint, side A)
//   [2]  mint_b (SPL mint, side B)
//   [3]  vault_a (PDA, will be created — owned by SPL Token program, ATA-style off-curve)
//   [4]  vault_b (PDA, will be created — same)
//   [5]  reserve_a_ct (signer, fresh keypair — becomes the side-A reserve ciphertext)
//   [6]  reserve_b_ct (signer, fresh keypair — becomes the side-B reserve ciphertext)
//   [7]  price_ct (signer, fresh keypair — public ciphertext for B-per-A*1e6)
//   [8]  admin (signer — captured into the pool, used later for SeedLiquidity)
//   [9]+ encrypt CPI accounts (encrypt_program, config, deposit, cpi_authority, caller_program,
//        network_encryption_key, payer, event_authority, system_program)
//
// For now, vault accounts are passed by the client (it creates them via SPL ATA helpers
// off-chain, owned by the pool PDA). On-chain we just record their addresses in the Pool
// struct and verify them at SeedLiquidity / Wrap time.
fn init_pool(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [pool_acct, mint_a, mint_b, vault_a, vault_b, ra_ct, rb_ct, price_ct, admin,
         encrypt_program, config, deposit, cpi_authority, caller_program,
         network_encryption_key, payer, event_authority, system_program, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if !admin.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (pool_bump, cpi_bump) = (data[0], data[1]);

    let bb = [pool_bump];
    let seeds = [Seed::from(b"dagon_pool" as &[u8]), Seed::from(&bb)];
    CreateAccount {
        from: payer, to: pool_acct, lamports: minimum_balance(Pool::LEN),
        space: Pool::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let ctx = EncryptContext {
        encrypt_program, config, deposit, cpi_authority, caller_program,
        network_encryption_key, payer, event_authority, system_program,
        cpi_authority_bump: cpi_bump,
    };
    ctx.create_plaintext_typed::<Uint128>(&0u128, ra_ct)?;
    ctx.create_plaintext_typed::<Uint128>(&0u128, rb_ct)?;
    ctx.create_plaintext_typed::<Uint128>(&0u128, price_ct)?;
    ctx.make_public(price_ct)?;

    let d = unsafe { pool_acct.borrow_unchecked_mut() };
    let pool = Pool::from_bytes_mut(d)?;
    pool.mint_a.copy_from_slice(mint_a.address().as_ref());
    pool.mint_b.copy_from_slice(mint_b.address().as_ref());
    pool.vault_a.copy_from_slice(vault_a.address().as_ref());
    pool.vault_b.copy_from_slice(vault_b.address().as_ref());
    pool.reserve_a.copy_from_slice(ra_ct.address().as_ref());
    pool.reserve_b.copy_from_slice(rb_ct.address().as_ref());
    pool.price_ct.copy_from_slice(price_ct.address().as_ref());
    pool.admin.copy_from_slice(admin.address().as_ref());
    pool.is_initialized = 1;
    pool.bump = pool_bump;
    Ok(())
}

// ── 1: SeedLiquidity ──
//
// Admin-only. Adds a plaintext amount to ONE side's reserve. Used at deploy time to seed the
// pool from the admin's SPL holdings. The amount is provided as an already-encrypted CT
// (admin encrypted it via gRPC with fhe_type=EUint128 authorized to this program).
//
// Account layout:
//   [0]  pool_acct
//   [1]  reserve_ct (the side being seeded — must equal pool.reserve_a or pool.reserve_b)
//   [2]  amount_ct (admin-encrypted EUint128 input, authorized to this program)
//   [3]  admin (signer)
//   [4]+ encrypt CPI accounts
//
// Data: [1, cpi_bump, side]  where side ∈ {0, 1}
fn seed_liquidity(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [pool_acct, reserve_ct, amount_ct, admin,
         encrypt_program, config, deposit, cpi_authority, caller_program,
         network_encryption_key, payer, event_authority, system_program, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !admin.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, side) = (data[0], data[1]);

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }
    if &pool.admin != admin.address().as_array() { return Err(ProgramError::InvalidArgument); }

    let expected_reserve: &[u8; 32] = match side {
        SIDE_A => &pool.reserve_a,
        SIDE_B => &pool.reserve_b,
        _ => return Err(ProgramError::InvalidArgument),
    };
    if reserve_ct.address().as_array() != expected_reserve {
        return Err(ProgramError::InvalidArgument);
    }

    let ctx = EncryptContext {
        encrypt_program, config, deposit, cpi_authority, caller_program,
        network_encryption_key, payer, event_authority, system_program,
        cpi_authority_bump: cpi_bump,
    };
    ctx.seed_reserve_graph(reserve_ct, amount_ct, reserve_ct)?;
    Ok(())
}

// ── 2: InitUserBalance ──
//
// Per-user, per-side. Creates the UserBalance PDA + its balance ciphertext.
//
// Account layout:
//   [0]  user_balance_acct (PDA, will be created)
//   [1]  pool_acct (read; just to require pool exists)
//   [2]  balance_ct (signer, fresh keypair — becomes the user's balance ciphertext)
//   [3]+ encrypt CPI accounts (payer = owner)
//
// Data: [2, ub_bump, cpi_bump, side]
fn init_user_balance(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [ub_acct, pool_acct, balance_ct,
         encrypt_program, config, deposit, cpi_authority, caller_program,
         network_encryption_key, owner, event_authority, system_program, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 3 { return Err(ProgramError::InvalidInstructionData); }
    let (ub_bump, cpi_bump, side) = (data[0], data[1], data[2]);
    if side != SIDE_A && side != SIDE_B {
        return Err(ProgramError::InvalidArgument);
    }

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }

    let side_byte = [side];
    let bb = [ub_bump];
    let seeds = [
        Seed::from(b"dagon_user" as &[u8]),
        Seed::from(owner.address().as_ref()),
        Seed::from(&side_byte),
        Seed::from(&bb),
    ];
    CreateAccount {
        from: owner, to: ub_acct, lamports: minimum_balance(UserBalance::LEN),
        space: UserBalance::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let ctx = EncryptContext {
        encrypt_program, config, deposit, cpi_authority, caller_program,
        network_encryption_key, payer: owner, event_authority, system_program,
        cpi_authority_bump: cpi_bump,
    };
    ctx.create_plaintext_typed::<Uint128>(&0u128, balance_ct)?;

    let d = unsafe { ub_acct.borrow_unchecked_mut() };
    let ub = UserBalance::from_bytes_mut(d)?;
    ub.owner.copy_from_slice(owner.address().as_ref());
    ub.balance_ct.copy_from_slice(balance_ct.address().as_ref());
    ub.side = side;
    ub.bump = ub_bump;
    Ok(())
}

// ── 3: WrapDeposit ──
//
// SPL → vault (plaintext SPL transfer) + balance += amount (FHE mint_to).
//
// Account layout:
//   [0]  pool_acct
//   [1]  user_balance_acct (read; verifies owner + side)
//   [2]  user_ata (SPL ATA holding the SPL the user is depositing)
//   [3]  vault_ata (SPL ATA owned by the pool PDA, holds the SPL backing for that side)
//   [4]  balance_ct (writable; balance += amount lands here)
//   [5]  amount_ct (user-encrypted EUint128 input, authorized to this program)
//   [6]+ encrypt CPI accounts (payer = owner)
//   [last] SPL token program
//
// Data: [3, cpi_bump, side, amount(u64 LE)]
//
// `amount` is the plaintext amount (SPL token units) — needs to match the encrypted amount_ct
// the user submitted. Like PC-Token's wrap, we trust the user to encode them consistently;
// FHE doesn't gate this. The plaintext is required for the SPL transfer.
fn wrap_deposit(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [pool_acct, ub_acct, user_ata, vault_ata, balance_ct, amount_ct,
         encrypt_program, config, deposit, cpi_authority, caller_program,
         network_encryption_key, owner, event_authority, system_program, _spl, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 10 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, side) = (data[0], data[1]);
    let amount = u64::from_le_bytes(data[2..10].try_into().unwrap());
    if amount == 0 { return Err(ProgramError::InvalidArgument); }

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }

    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub.side != side { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    // Vault must match the pool's recorded vault for this side.
    let expected_vault: &[u8; 32] = match side {
        SIDE_A => &pool.vault_a,
        SIDE_B => &pool.vault_b,
        _ => return Err(ProgramError::InvalidArgument),
    };
    if vault_ata.address().as_array() != expected_vault {
        return Err(ProgramError::InvalidArgument);
    }

    // Plaintext SPL transfer: user_ata → vault_ata.
    SplTransfer { from: user_ata, to: vault_ata, authority: owner, amount }.invoke()?;

    // FHE: balance += amount.
    let ctx = EncryptContext {
        encrypt_program, config, deposit, cpi_authority, caller_program,
        network_encryption_key, payer: owner, event_authority, system_program,
        cpi_authority_bump: cpi_bump,
    };
    ctx.mint_to_graph(balance_ct, amount_ct, balance_ct)?;
    Ok(())
}

// ── 4: SettleSwap ──
//
// The headline atomic op. One FHE graph spans pool reserves + user balances + price update.
// No chained CPIs, no copy_ciphertext, no multi-op atomicity assumption.
//
// Account layout:
//   [0]  pool_acct
//   [1]  user_in_balance_acct
//   [2]  user_out_balance_acct
//   [3]  reserve_in_ct (writable)
//   [4]  reserve_out_ct (writable)
//   [5]  user_in_balance_ct (writable)
//   [6]  user_out_balance_ct (writable)
//   [7]  amount_in_ct (input, user-encrypted, authorized to this program)
//   [8]  min_out_ct (input, user-encrypted, authorized to this program)
//   [9]  price_ct (writable)
//   [10]+ encrypt CPI accounts (payer = owner)
//
// Data: [4, cpi_bump, direction]   where direction: 0 = A→B, 1 = B→A
fn settle_swap(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [pool_acct, ub_in_acct, ub_out_acct,
         rin_ct, rout_ct, user_in_ct, user_out_ct,
         amount_in_ct, min_out_ct, price_ct,
         encrypt_program, config, deposit, cpi_authority, caller_program,
         network_encryption_key, owner, event_authority, system_program, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, direction) = (data[0], data[1]);

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }
    if price_ct.address().as_array() != &pool.price_ct {
        return Err(ProgramError::InvalidArgument);
    }

    // Resolve which reserve / which side based on direction.
    let (expected_rin, expected_rout, side_in, side_out) = match direction {
        0 => (&pool.reserve_a, &pool.reserve_b, SIDE_A, SIDE_B),
        1 => (&pool.reserve_b, &pool.reserve_a, SIDE_B, SIDE_A),
        _ => return Err(ProgramError::InvalidArgument),
    };
    if rin_ct.address().as_array() != expected_rin {
        return Err(ProgramError::InvalidArgument);
    }
    if rout_ct.address().as_array() != expected_rout {
        return Err(ProgramError::InvalidArgument);
    }

    // Verify user balances belong to the signer + match the right side.
    let ub_in_d = unsafe { ub_in_acct.borrow_unchecked() };
    let ub_in = UserBalance::from_bytes(ub_in_d)?;
    if &ub_in.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub_in.side != side_in { return Err(ProgramError::InvalidArgument); }
    if user_in_ct.address().as_array() != &ub_in.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let ub_out_d = unsafe { ub_out_acct.borrow_unchecked() };
    let ub_out = UserBalance::from_bytes(ub_out_d)?;
    if &ub_out.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub_out.side != side_out { return Err(ProgramError::InvalidArgument); }
    if user_out_ct.address().as_array() != &ub_out.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    // Run the single atomic graph.
    let ctx = EncryptContext {
        encrypt_program, config, deposit, cpi_authority, caller_program,
        network_encryption_key, payer: owner, event_authority, system_program,
        cpi_authority_bump: cpi_bump,
    };
    // atomic_swap_graph(rin, rout, user_in, user_out, amount_in, min_out, price)
    //   → (new_rin, new_rout, new_user_in, new_user_out, new_price)
    ctx.atomic_swap_graph(
        rin_ct, rout_ct, user_in_ct, user_out_ct, amount_in_ct, min_out_ct, price_ct,
        rin_ct, rout_ct, user_in_ct, user_out_ct, price_ct,
    )?;
    Ok(())
}

// ── 5: ViewBalance ──
//
// Decrypt-on-demand. Same pattern as PC-Token's view_balance: copy_ciphertext to a
// caller-provided ephemeral pubkey so the user can readCiphertext via gRPC.
//
// Account layout:
//   [0]  user_balance_acct
//   [1]  balance_ct (read-only source)
//   [2]  view_ct (signer, fresh keypair — becomes the new ciphertext)
//   [3]  new_authorized (the ephemeral pubkey that will read the view_ct)
//   [4]+ encrypt CPI accounts
fn view_balance(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [ub_acct, balance_ct, view_ct, new_authorized,
         encrypt_program, config, deposit, cpi_authority, caller_program,
         network_encryption_key, owner, event_authority, system_program, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];

    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let ctx = EncryptContext {
        encrypt_program, config, deposit, cpi_authority, caller_program,
        network_encryption_key, payer: owner, event_authority, system_program,
        cpi_authority_bump: cpi_bump,
    };
    ctx.copy_ciphertext(balance_ct, view_ct, new_authorized)?;
    Ok(())
}

// ── 6: AcceptDepositFromCold ──
//
// Consume an incoming pc-token-cold Ticket — created by the user's prior
// `pc_token_cold.transfer_out` call which rotated the CT's authority to dagon-points-pool's
// CPI authority. Read the ticket, verify signer == ticket.recipient, create the UserBalance
// struct referencing the now-pool-authorized CT, close the ticket (rent → user).
//
// The CT itself has its `authorized` field already rotated to point at dagon-points-pool's
// CPI authority — Encrypt enforces this at any future graph call. We don't need to verify
// the rotation on-chain here; if it didn't happen, the user can't use the resulting
// UserBalance for subsequent settle_swap (graph would reject) and they can recover by
// claiming the ticket back via pc-token-cold (if still alive).
//
// Account layout:
//   [0]  ticket_acct (writable; remains owned by pc-token-cold; lifecycle TBD)
//   [1]  ub_acct (PDA, will be created — dagon-pool UserBalance)
//   [2]  pool_acct (read; used to enforce pool-init invariant)
//   [3]  balance_ct (read; the now-pool-authorized CT)
//   [4]  owner (signer; must match ticket.recipient)
//   [5]  system_program (required by the inner CreateAccount CPI)
//
// Data: [6, ub_bump, side]
fn accept_deposit_from_cold(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [ticket_acct, ub_acct, pool_acct, balance_ct, owner, _sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (ub_bump, side) = (data[0], data[1]);
    if side != SIDE_A && side != SIDE_B { return Err(ProgramError::InvalidArgument); }

    // Verify the ticket lives in the trusted pc-token-cold program's namespace.
    if unsafe { ticket_acct.owner() } != &TRUSTED_COLD_PROGRAM_ID {
        return Err(ProgramError::InvalidArgument);
    }

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }

    // Parse the ticket (mirrored layout — keep TICKET_LEN in sync with pc-token-cold).
    let td = unsafe { ticket_acct.borrow_unchecked() };
    if td.len() < TICKET_LEN { return Err(ProgramError::InvalidAccountData); }
    let ticket_balance_ct = &td[TICKET_BALANCE_CT_OFFSET..TICKET_BALANCE_CT_OFFSET + 32];
    let ticket_recipient = &td[TICKET_RECIPIENT_OFFSET..TICKET_RECIPIENT_OFFSET + 32];
    let ticket_mint = &td[TICKET_MINT_OFFSET..TICKET_MINT_OFFSET + 32];

    if ticket_recipient != owner.address().as_ref() {
        return Err(ProgramError::InvalidArgument);
    }
    if ticket_balance_ct != balance_ct.address().as_ref() {
        return Err(ProgramError::InvalidArgument);
    }
    // Verify the ticket's mint matches the side this UserBalance is for.
    let expected_mint: &[u8; 32] = match side {
        SIDE_A => &pool.mint_a,
        SIDE_B => &pool.mint_b,
        _ => unreachable!(),
    };
    if ticket_mint != expected_mint { return Err(ProgramError::InvalidArgument); }

    // Create the UserBalance PDA referencing the rotated CT.
    let side_byte = [side];
    let bb = [ub_bump];
    let seeds = [
        Seed::from(b"dagon_user" as &[u8]),
        Seed::from(owner.address().as_ref()),
        Seed::from(&side_byte),
        Seed::from(&bb),
    ];
    CreateAccount {
        from: owner, to: ub_acct, lamports: minimum_balance(UserBalance::LEN),
        space: UserBalance::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;
    let d = unsafe { ub_acct.borrow_unchecked_mut() };
    let ub = UserBalance::from_bytes_mut(d)?;
    ub.owner.copy_from_slice(owner.address().as_ref());
    ub.balance_ct.copy_from_slice(balance_ct.address().as_ref());
    ub.side = side;
    ub.bump = ub_bump;

    // Close the ticket. We can't directly close an account owned by another program; we just
    // zero out the data and ask pc-token-cold to reclaim rent on its end via accept_in (if
    // pc-token-cold is the source). For tickets created by the OTHER direction (us
    // depositing into pc-token-cold), the ticket sits with us. Either way: best-effort —
    // the user can manually close stale tickets via pc-token-cold if needed.
    //
    // Pragmatically: do nothing here. Lifecycle of cross-program tickets is a known gap.
    let _ = ticket_acct;
    Ok(())
}

// ── 7: WithdrawToCold ──
//
// Rotate the user's hot UserBalance to pc-token-cold's CPI authority and CPI into
// pc-token-cold to write a matching Ticket. Close the UserBalance (rent → user).
//
// Two FHE-related operations happen:
//   1. transfer_ciphertext — rotates auth (likely metadata-only on pre-alpha; A10 in queue)
//   2. CPI into pc-token-cold's create_inbound_ticket_cpi — pure on-chain write, no FHE op
//
// Net: ONE FHE op (the rotate) + one regular CPI. Same TX, atomic.
//
// Account layout:
//   [0]  pool_acct (read)
//   [1]  ub_acct (writable; will be closed)
//   [2]  balance_ct (writable; transfer_ciphertext rewrites authorized field)
//   [3]  cold_cpi_authority (read; the new authorized — pc-token-cold's CPI authority PDA)
//   [4]  cold_program (read; pc-token-cold program account, used by the CPI invocation)
//   [5]  ticket_acct (writable; will be created by pc-token-cold's create_inbound_ticket_cpi)
//   [6]  mint (read; SPL mint for this side, passed through to the ticket)
//   [7]  owner (signer; must match UserBalance.owner — also the ticket recipient)
//   [8]+ encrypt CPI accounts (encrypt_program, config, deposit, cpi_authority, caller_program,
//        network_encryption_key, payer=owner, event_authority, system_program)
//
// Data: [7, cpi_bump, ticket_bump]
fn withdraw_to_cold(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [pool_acct, ub_acct, balance_ct, cold_cpi_authority, cold_program, ticket_acct, mint, owner,
         ep, cfg, dep, cpi_auth, caller, nk, payer, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, ticket_bump) = (data[0], data[1]);

    if cold_program.address() != &TRUSTED_COLD_PROGRAM_ID {
        return Err(ProgramError::InvalidArgument);
    }

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }

    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }
    let expected_mint: &[u8; 32] = match ub.side {
        SIDE_A => &pool.mint_a,
        SIDE_B => &pool.mint_b,
        _ => return Err(ProgramError::InvalidArgument),
    };
    if mint.address().as_array() != expected_mint {
        return Err(ProgramError::InvalidArgument);
    }

    // Step 1: rotate CT authority from us to pc-token-cold's CPI authority.
    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.transfer_ciphertext(balance_ct, cold_cpi_authority)?;

    // Step 2: CPI into pc-token-cold to create the matching Ticket. Account ordering must
    // match pc-token-cold's create_inbound_ticket_cpi expectations:
    //   [0] ticket_acct (writable)
    //   [1] balance_ct (read)
    //   [2] recipient (read; we pass `owner`)
    //   [3] mint (read)
    //   [4] payer (signer)
    //   [5] system_program
    let cpi_data = [PC_TOKEN_COLD_IX_CREATE_INBOUND_TICKET, ticket_bump];
    let cpi_accounts = [
        InstructionAccount { address: ticket_acct.address(), is_writable: true, is_signer: false },
        InstructionAccount { address: balance_ct.address(), is_writable: false, is_signer: false },
        InstructionAccount { address: owner.address(), is_writable: false, is_signer: false },
        InstructionAccount { address: mint.address(), is_writable: false, is_signer: false },
        InstructionAccount { address: payer.address(), is_writable: true, is_signer: true },
        InstructionAccount { address: sys.address(), is_writable: false, is_signer: false },
    ];
    let ix = InstructionView {
        program_id: &TRUSTED_COLD_PROGRAM_ID,
        data: &cpi_data,
        accounts: &cpi_accounts,
    };
    invoke_signed(&ix, &[ticket_acct, balance_ct, owner, mint, payer, sys, cold_program], &[])?;

    // Step 3: close the UserBalance — rent → owner.
    let lamports = ub_acct.lamports();
    ub_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + lamports);
    let ubd2 = unsafe { ub_acct.borrow_unchecked_mut() };
    for b in ubd2.iter_mut() { *b = 0; }
    Ok(())
}

// ── 8: MergeDepositIntoExisting ──
//
// Variant of accept_deposit_from_cold for the case where the user already has a UserBalance
// for this side. Instead of creating a new UserBalance, runs mint_to_graph to add the
// rotated cold balance INTO the existing hot balance:
//
//   existing_hot_balance += rotated_cold_balance
//
// The rotated CT becomes orphaned (rent stays). A future cleanup ix could close it; for now
// leaving it is harmless.
//
// Account layout:
//   [0]  ticket_acct (read; verifies provenance)
//   [1]  ub_acct (read; verifies signer + side)
//   [2]  pool_acct (read)
//   [3]  existing_balance_ct (writable; the user's existing hot balance_ct — receives the merge)
//   [4]  incoming_balance_ct (writable; the rotated cold CT — read by graph)
//   [5]  owner (signer; must match ticket.recipient and ub.owner)
//   [6]+ encrypt CPI accounts
//
// Data: [8, cpi_bump, side]
fn merge_deposit_into_existing(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [ticket_acct, ub_acct, pool_acct, existing_balance_ct, incoming_balance_ct, owner,
         ep, cfg, dep, cpi_auth, caller, nk, payer, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, side) = (data[0], data[1]);
    if side != SIDE_A && side != SIDE_B { return Err(ProgramError::InvalidArgument); }

    if unsafe { ticket_acct.owner() } != &TRUSTED_COLD_PROGRAM_ID {
        return Err(ProgramError::InvalidArgument);
    }

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }

    // Verify ticket bindings.
    let td = unsafe { ticket_acct.borrow_unchecked() };
    if td.len() < TICKET_LEN { return Err(ProgramError::InvalidAccountData); }
    let ticket_balance_ct = &td[TICKET_BALANCE_CT_OFFSET..TICKET_BALANCE_CT_OFFSET + 32];
    let ticket_recipient = &td[TICKET_RECIPIENT_OFFSET..TICKET_RECIPIENT_OFFSET + 32];
    let ticket_mint = &td[TICKET_MINT_OFFSET..TICKET_MINT_OFFSET + 32];

    if ticket_recipient != owner.address().as_ref() { return Err(ProgramError::InvalidArgument); }
    if ticket_balance_ct != incoming_balance_ct.address().as_ref() {
        return Err(ProgramError::InvalidArgument);
    }
    let expected_mint: &[u8; 32] = match side {
        SIDE_A => &pool.mint_a,
        SIDE_B => &pool.mint_b,
        _ => unreachable!(),
    };
    if ticket_mint != expected_mint { return Err(ProgramError::InvalidArgument); }

    // Verify existing UserBalance.
    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub.side != side { return Err(ProgramError::InvalidArgument); }
    if existing_balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    // Run merge graph: existing += incoming. Both CTs are dagon-points-pool authorized
    // (existing always was; incoming was rotated to us by the cold-side TransferOut).
    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.mint_to_graph(existing_balance_ct, incoming_balance_ct, existing_balance_ct)?;
    Ok(())
}

// ── 9: UnwrapBurn ──
//
// First step of hot → SPL unwrap. Burns `amount` from the user's encrypted hot UserBalance
// (FHE conditional: only if balance >= amount; no-op otherwise). Creates a WithdrawalReceipt
// recording the requested amount; the actual burned amount lands in `burned_ct`.
//
// Account layout:
//   [0]  pool_acct (read; resolves expected mint per side)
//   [1]  ub_acct (read; verifies signer + side)
//   [2]  receipt_acct (PDA, will be created at seeds [b"pool_receipt", burned_ct])
//   [3]  balance_ct (writable; FHE: -= amount on success)
//   [4]  amount_ct (read; user-encrypted EUint128 input)
//   [5]  burned_ct (writable; pre-allocated CT, FHE writes the actual burned amount)
//   [6]  mint (read; SPL mint identifier — must match pool.mint_a or pool.mint_b for side)
//   [7]+ encrypt CPI accounts (encrypt_program, config, deposit, cpi_authority, caller_program,
//        network_encryption_key, owner=signer, event_authority, system_program)
//
// Data: [9, receipt_bump, cpi_bump, side, amount(u64 LE)]
fn unwrap_burn(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [pool_acct, ub_acct, receipt_acct, balance_ct, amount_ct, burned_ct, mint,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 11 { return Err(ProgramError::InvalidInstructionData); }
    let (rb, cb, side) = (data[0], data[1], data[2]);
    let amount = u64::from_le_bytes(data[3..11].try_into().unwrap());
    if amount == 0 { return Err(ProgramError::InvalidArgument); }

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }
    let expected_mint: &[u8; 32] = match side {
        SIDE_A => &pool.mint_a,
        SIDE_B => &pool.mint_b,
        _ => return Err(ProgramError::InvalidArgument),
    };
    if mint.address().as_array() != expected_mint {
        return Err(ProgramError::InvalidArgument);
    }

    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub.side != side { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let bb = [rb];
    let seeds = [Seed::from(b"pool_receipt" as &[u8]), Seed::from(burned_ct.address().as_ref()), Seed::from(&bb)];
    CreateAccount {
        from: owner, to: receipt_acct, lamports: minimum_balance(WithdrawalReceipt::LEN),
        space: WithdrawalReceipt::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cb,
    };
    ctx.unwrap_burn_graph(balance_ct, amount_ct, balance_ct, burned_ct)?;

    let rd = unsafe { receipt_acct.borrow_unchecked_mut() };
    let r = WithdrawalReceipt::from_bytes_mut(rd)?;
    r.owner.copy_from_slice(owner.address().as_ref());
    r.amount = amount.to_le_bytes();
    r.pending_digest = [0u8; 32];
    r.mint.copy_from_slice(mint.address().as_ref());
    r.side = side;
    r.bump = rb;
    Ok(())
}

// ── 10: UnwrapDecrypt ──
//
// Second step. Calls Encrypt's request_decryption on burned_ct and stores the digest
// snapshot in the receipt. The executor will commit the decrypted plaintext into the
// request account asynchronously; unwrap_complete reads it after.
//
// Account layout:
//   [0]  receipt_acct (writable; digest gets written)
//   [1]  request_acct (signer, fresh keypair — the decryption result will land here)
//   [2]  burned_ct (read)
//   [3]+ encrypt CPI accounts
//
// Data: [10, cpi_bump]
fn unwrap_decrypt(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [receipt_acct, request_acct, burned_ct,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }

    let rd = unsafe { receipt_acct.borrow_unchecked() };
    let r = WithdrawalReceipt::from_bytes(rd)?;
    if owner.address().as_array() != &r.owner { return Err(ProgramError::InvalidArgument); }

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: data[0],
    };
    let digest = ctx.request_decryption(request_acct, burned_ct)?;

    let rd2 = unsafe { receipt_acct.borrow_unchecked_mut() };
    WithdrawalReceipt::from_bytes_mut(rd2)?.pending_digest = digest;
    Ok(())
}

// ── 11: UnwrapComplete ──
//
// Final step. Reads decrypted burned amount, verifies digest, CPIs into cold.release_spl
// (if the burn matched the requested amount) to drain SPL backing from cold's vault, then
// closes the satellite accounts: receipt, burned_ct (via close_ciphertext CPI), and the
// decryption request (via close_decryption_request CPI). All rent → owner.
//
// If the FHE burn no-op'd (insufficient balance), the cold.release_spl CPI is skipped but
// the closes still run — user reclaims rent and keeps their encrypted balance unchanged.
//
// Account layout:
//   [0]  receipt_acct (writable; closed last)
//   [1]  request_acct (writable; closed via CPI close_decryption_request)
//   [2]  owner (signer; rent destination)
//   [3]  mint (read)
//   [4]  burned_ct (writable; closed via CPI close_ciphertext)
//   [5]  cold_program (read; MUST equal TRUSTED_COLD_PROGRAM_ID)
//   [6]  cold_vault_pda (read)
//   [7]  cold_vault_ata (writable; SPL source)
//   [8]  user_ata (writable; SPL destination)
//   [9]  cold_release_auth (PDA at [b"cold_release"] under THIS program — signs the CPI)
//   [10] spl_token_program
//   [11..] encrypt CPI accounts: ep, cfg, dep, cpi_auth, caller, nk, evt, sys
//
// Data: [11, cpi_bump, cold_release_bump]
fn unwrap_complete(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [receipt_acct, request_acct, owner, mint, burned_ct,
         cold_program, cold_vault_pda, cold_vault_ata, user_ata, cold_release_auth, spl_token_program,
         ep, cfg, dep, cpi_auth, caller, nk, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, release_bump) = (data[0], data[1]);

    if cold_program.address() != &TRUSTED_COLD_PROGRAM_ID {
        return Err(ProgramError::InvalidArgument);
    }

    let rd = unsafe { receipt_acct.borrow_unchecked() };
    let r = WithdrawalReceipt::from_bytes(rd)?;
    if owner.address().as_array() != &r.owner { return Err(ProgramError::InvalidArgument); }
    if mint.address().as_array() != &r.mint { return Err(ProgramError::InvalidArgument); }
    let requested = r.requested_amount();
    let digest = r.pending_digest;

    let req_data = unsafe { request_acct.borrow_unchecked() };
    let burned: &u128 = accounts::read_decrypted_verified::<Uint128>(req_data, &digest)?;
    let burned_matches = *burned == requested as u128;

    if burned_matches {
        let amount_le = requested.to_le_bytes();
        let mut cpi_data = [0u8; 9];
        cpi_data[0] = PC_TOKEN_COLD_IX_RELEASE_SPL;
        cpi_data[1..9].copy_from_slice(&amount_le);

        let cpi_accounts = [
            InstructionAccount { address: cold_vault_pda.address(), is_writable: false, is_signer: false },
            InstructionAccount { address: cold_vault_ata.address(), is_writable: true, is_signer: false },
            InstructionAccount { address: user_ata.address(), is_writable: true, is_signer: false },
            InstructionAccount { address: mint.address(), is_writable: false, is_signer: false },
            InstructionAccount { address: cold_release_auth.address(), is_writable: false, is_signer: true },
            InstructionAccount { address: spl_token_program.address(), is_writable: false, is_signer: false },
        ];
        let ix = InstructionView {
            program_id: &TRUSTED_COLD_PROGRAM_ID,
            data: &cpi_data,
            accounts: &cpi_accounts,
        };
        let bb = [release_bump];
        let release_seeds = [Seed::from(b"cold_release" as &[u8]), Seed::from(&bb)];
        invoke_signed(
            &ix,
            &[
                cold_vault_pda, cold_vault_ata, user_ata, mint,
                cold_release_auth, spl_token_program, cold_program,
            ],
            &[Signer::from(&release_seeds)],
        )?;
    }

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.close_ciphertext(burned_ct, owner)?;
    ctx.close_decryption_request(request_acct, owner)?;

    let rl = receipt_acct.lamports();
    receipt_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + rl);
    let rd2 = unsafe { receipt_acct.borrow_unchecked_mut() };
    for b in rd2.iter_mut() { *b = 0; }
    Ok(())
}

// ── 12: MigrateVaultToCold ──
//
// One-time admin migration of legacy SPL backing from this program's pool.vault_x into
// cold's per-mint vault. After running for both sides, all SPL custody lives in cold and
// pool.vault_x is empty. Idempotent — re-running with an already-empty pool.vault_x is a
// no-op success. SPL transfer signed by the pool PDA.
//
// Account layout:
//   [0]  pool_acct (read; verifies admin + recorded vaults)
//   [1]  pool_vault_ata (writable; SPL source — owned by pool PDA)
//   [2]  cold_vault_ata (writable; SPL destination — cold's per-mint vault ATA)
//   [3]  admin (signer; must match pool.admin)
//   [4]  spl_token_program
//
// Data: [12, side, pool_pda_bump]
fn migrate_vault_to_cold(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [pool_acct, pool_vault_ata, cold_vault_ata, admin, _spl, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !admin.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (side, pool_bump) = (data[0], data[1]);

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }
    if &pool.admin != admin.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if pool.bump != pool_bump { return Err(ProgramError::InvalidArgument); }

    let expected_pool_vault: &[u8; 32] = match side {
        SIDE_A => &pool.vault_a,
        SIDE_B => &pool.vault_b,
        _ => return Err(ProgramError::InvalidArgument),
    };
    if pool_vault_ata.address().as_array() != expected_pool_vault {
        return Err(ProgramError::InvalidArgument);
    }

    let pvad = unsafe { pool_vault_ata.borrow_unchecked() };
    if pvad.len() < 72 { return Err(ProgramError::InvalidAccountData); }
    let amount = u64::from_le_bytes(pvad[64..72].try_into().unwrap());
    if amount == 0 { return Ok(()); }

    let bb = [pool_bump];
    let seeds = [Seed::from(b"dagon_pool" as &[u8]), Seed::from(&bb)];
    SplTransfer { from: pool_vault_ata, to: cold_vault_ata, authority: pool_acct, amount }
        .invoke_signed(&[Signer::from(&seeds)])?;
    Ok(())
}

// ── 13: CloseBurn ──
//
// First step of hot UB teardown. Drains the encrypted hot balance to 0 and writes the
// pre-burn amount into burned_ct. Mirrors cold's close_burn shape.
//
// Account layout:
//   [0]  pool_acct (read; mint check)
//   [1]  ub_acct (read; verify owner + side)
//   [2]  receipt_acct (PDA at [b"pool_receipt", burned_ct])
//   [3]  balance_ct (writable; FHE: -> 0)
//   [4]  burned_ct (writable; pre-allocated CT, FHE writes pre-burn balance)
//   [5]  mint (read)
//   [6]+ encrypt CPI accounts
//
// Data: [13, receipt_bump, cpi_bump, side]
fn close_burn(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [pool_acct, ub_acct, receipt_acct, balance_ct, burned_ct, mint,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 3 { return Err(ProgramError::InvalidInstructionData); }
    let (rb, cb, side) = (data[0], data[1], data[2]);

    let pd = unsafe { pool_acct.borrow_unchecked() };
    let pool = Pool::from_bytes(pd)?;
    if pool.is_initialized != 1 { return Err(ProgramError::UninitializedAccount); }
    let expected_mint: &[u8; 32] = match side {
        SIDE_A => &pool.mint_a,
        SIDE_B => &pool.mint_b,
        _ => return Err(ProgramError::InvalidArgument),
    };
    if mint.address().as_array() != expected_mint {
        return Err(ProgramError::InvalidArgument);
    }

    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub.side != side { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let bb = [rb];
    let seeds = [Seed::from(b"pool_receipt" as &[u8]), Seed::from(burned_ct.address().as_ref()), Seed::from(&bb)];
    CreateAccount {
        from: owner, to: receipt_acct, lamports: minimum_balance(WithdrawalReceipt::LEN),
        space: WithdrawalReceipt::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cb,
    };
    ctx.close_burn_graph(balance_ct, balance_ct, burned_ct)?;

    let rd = unsafe { receipt_acct.borrow_unchecked_mut() };
    let r = WithdrawalReceipt::from_bytes_mut(rd)?;
    r.owner.copy_from_slice(owner.address().as_ref());
    r.amount = 0u64.to_le_bytes();
    r.pending_digest = [0u8; 32];
    r.mint.copy_from_slice(mint.address().as_ref());
    r.side = side;
    r.bump = rb;
    Ok(())
}

// ── 14: CloseComplete ──
//
// Final step of hot UB teardown. Reads decrypted burned amount, CPIs into cold.release_spl
// to drain the corresponding SPL backing from cold's vault, then closes the UB + balance_ct
// + burned_ct + decryption request + receipt. All rent → owner.
//
// Account layout:
//   [0]  receipt_acct (writable; closed last)
//   [1]  request_acct (writable; closed via CPI close_decryption_request)
//   [2]  owner (signer; rent destination)
//   [3]  mint (read)
//   [4]  ub_acct (writable; zeroed locally)
//   [5]  balance_ct (writable; closed via CPI close_ciphertext)
//   [6]  burned_ct (writable; closed via CPI close_ciphertext)
//   [7]  cold_program (read; CPI target — must equal TRUSTED_COLD_PROGRAM_ID)
//   [8]  cold_vault_pda (read)
//   [9]  cold_vault_ata (writable; SPL source)
//   [10] user_ata (writable; SPL destination)
//   [11] cold_release_auth (PDA at [b"cold_release"] under THIS program — signs the CPI)
//   [12] spl_token_program
//   [13..] encrypt CPI accounts: ep, cfg, dep, cpi_auth, caller, nk, evt, sys
//
// Data: [14, cpi_bump, cold_release_bump]
fn close_complete(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [receipt_acct, request_acct, owner, mint, ub_acct, balance_ct, burned_ct,
         cold_program, cold_vault_pda, cold_vault_ata, user_ata, cold_release_auth, spl_token_program,
         ep, cfg, dep, cpi_auth, caller, nk, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, release_bump) = (data[0], data[1]);

    if cold_program.address() != &TRUSTED_COLD_PROGRAM_ID {
        return Err(ProgramError::InvalidArgument);
    }

    let rd = unsafe { receipt_acct.borrow_unchecked() };
    let r = WithdrawalReceipt::from_bytes(rd)?;
    if owner.address().as_array() != &r.owner { return Err(ProgramError::InvalidArgument); }
    if mint.address().as_array() != &r.mint { return Err(ProgramError::InvalidArgument); }
    let digest = r.pending_digest;
    let side = r.side;

    let ubd = unsafe { ub_acct.borrow_unchecked() };
    let ub = UserBalance::from_bytes(ubd)?;
    if &ub.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if ub.side != side { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ub.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let req_data = unsafe { request_acct.borrow_unchecked() };
    let burned: &u128 = accounts::read_decrypted_verified::<Uint128>(req_data, &digest)?;
    let release_amount = *burned as u64;

    if release_amount > 0 {
        let amount_le = release_amount.to_le_bytes();
        let mut cpi_data = [0u8; 9];
        cpi_data[0] = PC_TOKEN_COLD_IX_RELEASE_SPL;
        cpi_data[1..9].copy_from_slice(&amount_le);
        let cpi_accounts = [
            InstructionAccount { address: cold_vault_pda.address(), is_writable: false, is_signer: false },
            InstructionAccount { address: cold_vault_ata.address(), is_writable: true, is_signer: false },
            InstructionAccount { address: user_ata.address(), is_writable: true, is_signer: false },
            InstructionAccount { address: mint.address(), is_writable: false, is_signer: false },
            InstructionAccount { address: cold_release_auth.address(), is_writable: false, is_signer: true },
            InstructionAccount { address: spl_token_program.address(), is_writable: false, is_signer: false },
        ];
        let ix = InstructionView {
            program_id: &TRUSTED_COLD_PROGRAM_ID,
            data: &cpi_data,
            accounts: &cpi_accounts,
        };
        let bb = [release_bump];
        let release_seeds = [Seed::from(b"cold_release" as &[u8]), Seed::from(&bb)];
        invoke_signed(
            &ix,
            &[
                cold_vault_pda, cold_vault_ata, user_ata, mint,
                cold_release_auth, spl_token_program, cold_program,
            ],
            &[Signer::from(&release_seeds)],
        )?;
    }

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.close_ciphertext(balance_ct, owner)?;
    ctx.close_ciphertext(burned_ct, owner)?;
    ctx.close_decryption_request(request_acct, owner)?;

    let ubl = ub_acct.lamports();
    ub_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + ubl);
    let ubd2 = unsafe { ub_acct.borrow_unchecked_mut() };
    for b in ubd2.iter_mut() { *b = 0; }

    let rl = receipt_acct.lamports();
    receipt_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + rl);
    let rd2 = unsafe { receipt_acct.borrow_unchecked_mut() };
    for b in rd2.iter_mut() { *b = 0; }
    Ok(())
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use encrypt_types::graph::{get_node, parse_graph, GraphNodeKind};
    use encrypt_types::identifier::*;
    use encrypt_types::types::FheType;
    use super::{atomic_swap_graph, mint_to_graph, seed_reserve_graph};

    fn run_mock(graph_fn: fn() -> Vec<u8>, inputs: &[u128], fhe_types: &[FheType]) -> Vec<u128> {
        let data = graph_fn();
        let pg = parse_graph(&data).unwrap();
        let num = pg.header().num_nodes() as usize;
        let mut digests: Vec<[u8; 32]> = Vec::with_capacity(num);
        let mut inp = 0usize;
        for i in 0..num {
            let n = get_node(pg.node_bytes(), i as u16).unwrap();
            let ft = FheType::from_u8(n.fhe_type()).unwrap_or(FheType::EUint128);
            let d = match n.kind() {
                k if k == GraphNodeKind::Input as u8 => {
                    let v = inputs[inp]; let t = fhe_types[inp]; inp += 1;
                    encode_mock_digest(t, v)
                }
                k if k == GraphNodeKind::Constant as u8 => {
                    let bw = ft.byte_width().min(16);
                    let off = n.const_offset() as usize;
                    let mut buf = [0u8; 16];
                    buf[..bw].copy_from_slice(&pg.constants()[off..off + bw]);
                    encode_mock_digest(ft, u128::from_le_bytes(buf))
                }
                k if k == GraphNodeKind::Op as u8 => {
                    let (a, b, c) = (n.input_a() as usize, n.input_b() as usize, n.input_c() as usize);
                    if n.op_type() == 60 { mock_select(&digests[a], &digests[b], &digests[c]) }
                    else if b == 0xFFFF {
                        mock_unary_compute(unsafe { core::mem::transmute(n.op_type()) }, &digests[a], ft)
                    } else {
                        mock_binary_compute(unsafe { core::mem::transmute(n.op_type()) }, &digests[a], &digests[b], ft)
                    }
                }
                k if k == GraphNodeKind::Output as u8 => digests[n.input_a() as usize],
                _ => panic!("bad node"),
            };
            digests.push(d);
        }
        (0..num).filter(|&i| get_node(pg.node_bytes(), i as u16).unwrap().kind() == GraphNodeKind::Output as u8)
            .map(|i| decode_mock_identifier(&digests[i])).collect()
    }

    const T: FheType = FheType::EUint128;

    #[test] fn mint_to() { let r = run_mock(mint_to_graph, &[500, 300], &[T, T]); assert_eq!(r[0], 800); }
    #[test] fn seed_reserve() { let r = run_mock(seed_reserve_graph, &[1000, 500], &[T, T]); assert_eq!(r[0], 1500); }

    // ── Atomic swap: happy path ──
    #[test] fn atomic_swap_happy() {
        // reserves: 1000 USDC, 100 SOL (synthetic units — pool ratio).
        // user_in (USDC): 500, user_out (SOL): 0
        // amount_in = 100, min_out = 0, price = 100_000 (0.1 in 1e6 units)
        let r = run_mock(
            atomic_swap_graph,
            &[1000, 100, 500, 0, 100, 0, 100_000],
            &[T, T, T, T, T, T, T],
        );
        // (final_reserve_in, final_reserve_out, final_user_in, final_user_out, final_price)
        assert_eq!(r[0], 1100, "reserve_in += amount_in");
        assert!(r[1] < 100 && r[1] > 0, "reserve_out decreased");
        assert_eq!(r[2], 400, "user_in -= amount_in");
        assert!(r[3] > 0, "user_out credited");
        assert!(r[4] > 0, "price updated");
    }

    // ── User balance insufficient: full no-op ──
    #[test] fn atomic_swap_user_insufficient() {
        // user_in = 50, amount_in = 100 → can't afford.
        let r = run_mock(
            atomic_swap_graph,
            &[1000, 100, 50, 0, 100, 0, 100_000],
            &[T, T, T, T, T, T, T],
        );
        assert_eq!(r[0], 1000, "reserve_in unchanged");
        assert_eq!(r[1], 100, "reserve_out unchanged");
        assert_eq!(r[2], 50, "user_in unchanged");
        assert_eq!(r[3], 0, "user_out unchanged");
        assert_eq!(r[4], 100_000, "price unchanged");
    }

    // ── Slippage: full no-op ──
    #[test] fn atomic_swap_slippage() {
        // min_out is way too high.
        let r = run_mock(
            atomic_swap_graph,
            &[1000, 100, 500, 0, 100, 999, 100_000],
            &[T, T, T, T, T, T, T],
        );
        assert_eq!(r[0], 1000, "reserve_in unchanged");
        assert_eq!(r[2], 500, "user_in unchanged");
        assert_eq!(r[3], 0, "user_out unchanged");
    }

    // ── k-invariant preserved across runs ──
    #[test] fn atomic_swap_k_preserved() {
        let mut ra = 100_000u128;
        let mut rb = 100_000u128;
        let mut ua = 50_000u128;
        let mut ub = 50_000u128;
        let initial_k = ra * rb;
        let mut price = 1_000_000u128;
        for _ in 0..5 {
            let r = run_mock(
                atomic_swap_graph,
                &[ra, rb, ua, ub, 100, 0, price],
                &[T, T, T, T, T, T, T],
            );
            ra = r[0]; rb = r[1]; ua = r[2]; ub = r[3]; price = r[4];
        }
        assert!(ra * rb >= initial_k, "k invariant preserved");
    }

    #[test] fn graph_shapes() {
        swap_shapes(mint_to_graph, 2, 1);
        swap_shapes(seed_reserve_graph, 2, 1);
        swap_shapes(atomic_swap_graph, 7, 5);
    }
    fn swap_shapes(f: fn() -> Vec<u8>, ni: u8, no: u8) {
        let g = f(); let pg = parse_graph(&g).unwrap();
        assert_eq!(pg.header().num_inputs(), ni);
        assert_eq!(pg.header().num_outputs(), no);
    }
}
