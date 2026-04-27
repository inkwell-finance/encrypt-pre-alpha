// Copyright (c) Inkwell Finance.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#![allow(unexpected_cfgs)]

//! # pc-token-cold
//!
//! Universal cold-storage balance program. Holds a user's encrypted token balances under one
//! Encrypt CPI authority and supports plain transfers between any two `BalanceAccount`s
//! (user→user, user→program PDA, program→user) as a single FHE graph.
//!
//! Designed as the cold-storage tier in the two-tier composability model documented in
//! ADR-021. Each protocol that wants atomic multi-leg ops keeps its own per-protocol
//! "hot tier" (e.g. `dagon-points-pool`); users move balances cold↔hot via
//! `transfer_ciphertext` rotations bound by `Ticket` accounts.
//!
//! ## Why "cold storage"
//!
//! Atomic protocol-specific math (AMM constant-product, lending interest, perp PnL) lives
//! in each protocol's own program because a single FHE graph can only span ciphertexts under
//! one CPI authority. pc-token-cold can't host that math without becoming a settlement-hub
//! monolith. Instead it provides the universal *transfer* surface: deposit cold→hot before
//! using a protocol, withdraw hot→cold after, and any other balance moves (P2P, multi-protocol
//! routing) happen here as single-graph transfers.
//!
//! ## Instructions
//!
//! 0. `InitVault` — per-mint vault (anyone, idempotent)
//! 1. `InitBalance` — per (user, mint) BalanceAccount (user signs)
//! 2. `Wrap` — SPL → user balance (single graph)
//! 3. `Transfer` — atomic user→user-or-program transfer (single graph)
//! 4. `ViewBalance` — copy_ciphertext for client decrypt-on-demand
//! 5. `TransferOut` — rotate user's balance authority to another program + write a Ticket
//! 6. `AcceptIn` — consume an incoming Ticket + create a BalanceAccount
//! 7. `CreateInboundTicketCpi` — permissionless helper. Other programs CPI in to write a
//!    Ticket targeting pc-token-cold after they've done their own `transfer_ciphertext`
//!    rotation.

use encrypt_dsl::prelude::encrypt_fn;
use encrypt_pinocchio::accounts;
use encrypt_pinocchio::EncryptContext;
use encrypt_types::encrypted::{EUint128, Uint128};
use pinocchio::{
    cpi::{Seed, Signer},
    entrypoint,
    error::ProgramError,
    AccountView, Address, ProgramResult,
};
use pinocchio_system::instructions::CreateAccount;
use pinocchio_token::instructions::Transfer as SplTransfer;

entrypoint!(process_instruction);

pub const ID: Address = Address::new_from_array([8u8; 32]);

/// Whitelisted minter program — currently just `dagon-points-pool`
/// (`2RUL5771DW7QnVg82sM5p3SbfEhuFz2ViFEJ49UYP6D7` on Solana devnet). When dagon-pool's
/// `unwrap_complete` CPIs into `release_spl`, it signs a PDA at seeds `[b"cold_release"]`
/// under this program ID. cold verifies that signer matches the precomputed authority below.
///
/// **If you redeploy dagon-points-pool, both `TRUSTED_DAGON_POOL_ID` and
/// `TRUSTED_DAGON_POOL_RELEASE_AUTHORITY` MUST be regenerated.** Use:
///   PublicKey.findProgramAddressSync([Buffer.from("cold_release")], <new pool id>)
pub const TRUSTED_DAGON_POOL_ID: Address = Address::new_from_array([
    0x15, 0x20, 0x86, 0xf1, 0x9d, 0x1a, 0xc9, 0xab, 0x3b, 0x28, 0x5b, 0x65, 0x4f, 0x90, 0xa5, 0xd4,
    0xbc, 0x4e, 0x0e, 0xfd, 0x63, 0x87, 0xfa, 0x22, 0x7d, 0x2e, 0x3d, 0xbc, 0x9f, 0x26, 0x75, 0x32,
]);

/// Precomputed PDA at seeds `[b"cold_release"]` under `TRUSTED_DAGON_POOL_ID`. Pinned here
/// so `release_spl` can do a 32-byte equality check instead of a runtime Sha256 derivation
/// per call. Bump = 255.
pub const TRUSTED_DAGON_POOL_RELEASE_AUTHORITY: [u8; 32] = [
    0x0f, 0x52, 0x7f, 0xf2, 0x27, 0x9e, 0xa5, 0x7e, 0xd4, 0x0e, 0xac, 0xfc, 0x4f, 0x4d, 0xc9, 0x92,
    0xe9, 0x21, 0xcf, 0x2f, 0xea, 0x60, 0xef, 0xc8, 0x20, 0xae, 0xd2, 0xf4, 0x3c, 0x14, 0x8a, 0x2a,
];

// ── Account layouts ──

/// Vault — per-mint PDA at seeds `[b"pc_vault", mint]`. Owns the SPL ATA holding the
/// backing tokens for all wrapped balances of `mint`.
#[repr(C)]
pub struct Vault {
    pub mint: [u8; 32],
    pub bump: u8,
    pub _pad: [u8; 7],
}

impl Vault {
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

/// BalanceAccount — per (owner, mint) PDA at seeds `[b"pc_balance", owner, mint]`.
/// `owner` is whoever signs to use this balance — typically a user wallet, but a program
/// PDA also works (programs can hold balances for protocol-internal accounting).
#[repr(C)]
pub struct BalanceAccount {
    pub owner: [u8; 32],
    pub mint: [u8; 32],
    pub balance_ct: [u8; 32],
    pub bump: u8,
    pub _pad: [u8; 7],
}

impl BalanceAccount {
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

/// Ticket — pending transfer record. Created by either side of a cold↔hot rotation:
///   - cold→hot: pc-token-cold's `TransferOut` ix creates one targeting another program
///   - hot→cold: another program calls `CreateInboundTicketCpi` after rotating auth back to pc-token-cold
///
/// `recipient` binds the ticket to a specific user; only that user's signature can claim
/// the corresponding balance via `AcceptIn`. Without this binding any observer who saw
/// the rotation could try to claim the CT.
///
/// PDA seeds: `[b"pc_ticket", balance_ct]` — one ticket per pending CT. Tickets are
/// always closable by the recipient (claims it via AcceptIn) — rent flows to the
/// recipient.
#[repr(C)]
pub struct Ticket {
    pub balance_ct: [u8; 32],
    pub recipient: [u8; 32],
    pub mint: [u8; 32],
    pub bump: u8,
    pub _pad: [u8; 7],
}

impl Ticket {
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

/// Trusted-minter registry. Singleton PDA at seeds `[b"cold_minters"]`. `release_spl`
/// authorizes a CPI caller by checking that their release-authority PDA address is in
/// `authorities[..len]`. Admin-gated add/remove. Capped at `MAX_TRUSTED_MINTERS` to keep
/// this struct fixed-size — a future migration can lift the cap if more minters come on.
///
/// The hardcoded `TRUSTED_DAGON_POOL_RELEASE_AUTHORITY` is also accepted by `release_spl`
/// (bootstrap entry) so the registry is optional during the migration window. After
/// stable, the constant can be removed and the registry is the sole source of truth.
pub const MAX_TRUSTED_MINTERS: usize = 8;

#[repr(C)]
pub struct MintersRegistry {
    pub admin: [u8; 32],
    pub bump: u8,
    pub len: u8,
    pub _pad: [u8; 6],
    pub authorities: [[u8; 32]; MAX_TRUSTED_MINTERS],
}

impl MintersRegistry {
    pub const LEN: usize = core::mem::size_of::<Self>();
    pub fn from_bytes(d: &[u8]) -> Result<&Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &*(d.as_ptr() as *const Self) })
    }
    pub fn from_bytes_mut(d: &mut [u8]) -> Result<&mut Self, ProgramError> {
        if d.len() < Self::LEN { return Err(ProgramError::InvalidAccountData); }
        Ok(unsafe { &mut *(d.as_mut_ptr() as *mut Self) })
    }
    pub fn contains(&self, addr: &[u8; 32]) -> bool {
        let n = (self.len as usize).min(MAX_TRUSTED_MINTERS);
        self.authorities[..n].iter().any(|a| a == addr)
    }
    pub fn position(&self, addr: &[u8; 32]) -> Option<usize> {
        let n = (self.len as usize).min(MAX_TRUSTED_MINTERS);
        self.authorities[..n].iter().position(|a| a == addr)
    }
}

/// Temporary receipt for unwrap. Created by `unwrap_burn`, populated with a digest by
/// `unwrap_decrypt`, consumed + closed by `unwrap_complete`.
/// PDA seeds: `[b"pc_receipt", burned_ct]`.
#[repr(C)]
pub struct WithdrawalReceipt {
    pub owner: [u8; 32],
    pub amount: [u8; 8],           // requested plaintext amount (u64, SPL-bound)
    pub pending_digest: [u8; 32],  // digest snapshot taken at request_decryption
    pub mint: [u8; 32],            // SPL mint to release on complete
    pub bump: u8,
    pub _pad: [u8; 7],
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

/// `balance += amount`. Used by Wrap (SPL → balance).
#[encrypt_fn]
fn mint_to_graph(balance: EUint128, amount: EUint128) -> EUint128 {
    balance + amount
}

/// Atomic transfer between two BalanceAccounts.
/// `valid = from_balance >= amount`. Outputs new balances; no-ops on insufficient.
#[encrypt_fn]
fn transfer_graph(
    from_balance: EUint128,
    to_balance: EUint128,
    amount: EUint128,
) -> (EUint128, EUint128) {
    let sufficient = from_balance >= amount;
    let new_from = if sufficient { from_balance - amount } else { from_balance };
    let new_to = if sufficient { to_balance + amount } else { to_balance };
    (new_from, new_to)
}

/// Conditional burn for unwrap. `burned = amount if sufficient else 0`. The decrypted
/// burned value is used by `unwrap_complete` to gate the SPL release.
#[encrypt_fn]
fn unwrap_burn_graph(balance: EUint128, amount: EUint128) -> (EUint128, EUint128) {
    let s = balance >= amount;
    let new_balance = if s { balance - amount } else { balance };
    let burned = if s { amount } else { amount - amount };
    (new_balance, burned)
}

/// Unconditional drain-all burn for `close_balance`. Always sets balance to 0 and outputs
/// the pre-burn balance as `burned`. close_complete uses the decrypted `burned` to release
/// the corresponding SPL backing, then closes the BA + balance_ct + burned_ct + receipt +
/// request_acct so the user reclaims all stranded rent in one flow.
#[encrypt_fn]
fn close_burn_graph(balance: EUint128) -> (EUint128, EUint128) {
    let zero = balance - balance;
    (zero, balance)
}

// ── Dispatch ──

fn process_instruction(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    match data.split_first() {
        Some((&0, rest)) => init_vault(program_id, accounts, rest),
        Some((&1, rest)) => init_balance(program_id, accounts, rest),
        Some((&2, rest)) => wrap(accounts, rest),
        Some((&3, rest)) => transfer(accounts, rest),
        Some((&4, rest)) => view_balance(accounts, rest),
        Some((&5, rest)) => transfer_out(program_id, accounts, rest),
        Some((&6, rest)) => accept_in(program_id, accounts, rest),
        Some((&7, rest)) => create_inbound_ticket_cpi(program_id, accounts, rest),
        Some((&8, rest)) => merge_in(accounts, rest),
        // 9-11: unwrap flow (cold → SPL). Same multi-step pattern as PC-Token: burn (FHE),
        // decrypt (executor), complete (verify + SPL release).
        Some((&9, rest)) => unwrap_burn(program_id, accounts, rest),
        Some((&10, rest)) => unwrap_decrypt(accounts, rest),
        Some((&11, rest)) => unwrap_complete(accounts, rest),
        // 12: settlement-program escape hatch. Whitelisted minters CPI in to release SPL
        // backing on behalf of an FHE-burn they performed in their own program.
        Some((&12, rest)) => release_spl(accounts, rest),
        // 13-14: drain-all + close. Reuses unwrap_decrypt (ix=10) for the middle step.
        Some((&13, rest)) => close_burn(program_id, accounts, rest),
        Some((&14, rest)) => close_complete(accounts, rest),
        // 15-16: trusted-minter registry config PDA. Admin-gated add/remove of release
        // authorities used by `release_spl`. Migrates beyond the single hardcoded minter.
        Some((&15, rest)) => init_minters_registry(program_id, accounts, rest),
        Some((&16, rest)) => set_trusted_minter(accounts, rest),
        _ => Err(ProgramError::InvalidInstructionData),
    }
}

// ── 0: InitVault ──
//
// Account layout:
//   [0] vault_pda (PDA, will be created)
//   [1] mint
//   [2] payer (signer)
//   [3] system_program
//
// Data: [0, vault_bump]
fn init_vault(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [vault_pda, mint, payer, _sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let vb = data[0];

    let bb = [vb];
    let seeds = [Seed::from(b"pc_vault" as &[u8]), Seed::from(mint.address().as_ref()), Seed::from(&bb)];
    CreateAccount {
        from: payer, to: vault_pda, lamports: minimum_balance(Vault::LEN),
        space: Vault::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;
    let d = unsafe { vault_pda.borrow_unchecked_mut() };
    let v = Vault::from_bytes_mut(d)?;
    v.mint.copy_from_slice(mint.address().as_ref());
    v.bump = vb;
    Ok(())
}

// ── 1: InitBalance ──
//
// Permissionless: the `owner` does NOT need to sign. Anyone willing to pay rent can create
// a BalanceAccount for any wallet — the BA is owned by this program and grants no rights to
// the creator (only `owner`'s signature can later authorize transfers/wraps from it). This
// enables sender-funded recipient onboarding for U2U transfers (one TX: create recipient BA
// + transfer), while harmless for everyone else (creating an empty 0-balance BA for someone
// just costs the creator some rent).
//
// Account layout:
//   [0] balance_acct (PDA, will be created)
//   [1] owner (any pubkey — does NOT need to sign; PDAs work too if a program CPIs in)
//   [2] mint
//   [3] balance_ct (signer, fresh keypair)
//   [4]+ encrypt CPI accounts (encrypt_program, config, deposit, cpi_authority, caller_program,
//        network_encryption_key, payer, event_authority, system_program)
//
// Data: [1, ba_bump, cpi_bump]
fn init_balance(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [ba_acct, owner, mint, balance_ct,
         ep, cfg, dep, cpi_auth, caller, nk, payer, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (ba_bump, cpi_bump) = (data[0], data[1]);

    let bb = [ba_bump];
    let seeds = [
        Seed::from(b"pc_balance" as &[u8]),
        Seed::from(owner.address().as_ref()),
        Seed::from(mint.address().as_ref()),
        Seed::from(&bb),
    ];
    CreateAccount {
        from: payer, to: ba_acct, lamports: minimum_balance(BalanceAccount::LEN),
        space: BalanceAccount::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.create_plaintext_typed::<Uint128>(&0u128, balance_ct)?;

    let d = unsafe { ba_acct.borrow_unchecked_mut() };
    let ba = BalanceAccount::from_bytes_mut(d)?;
    ba.owner.copy_from_slice(owner.address().as_ref());
    ba.mint.copy_from_slice(mint.address().as_ref());
    ba.balance_ct.copy_from_slice(balance_ct.address().as_ref());
    ba.bump = ba_bump;
    Ok(())
}

// ── 2: Wrap ──
//
// Account layout:
//   [0] vault_pda (read; for spl_mint check)
//   [1] balance_acct (read; verify owner+mint)
//   [2] user_ata (writable; SPL source)
//   [3] vault_ata (writable; SPL destination — owned by vault_pda)
//   [4] balance_ct (writable; FHE op writes here)
//   [5] amount_ct (input)
//   [6]+ encrypt CPI accounts (payer = owner)
//   [last] SPL token program
//
// Data: [2, cpi_bump, amount(u64 LE)]
fn wrap(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [vault_pda, ba_acct, user_ata, vault_ata, balance_ct, amount_ct,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, _spl, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 9 { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];
    let amount = u64::from_le_bytes(data[1..9].try_into().unwrap());
    if amount == 0 { return Err(ProgramError::InvalidArgument); }

    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ba.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let vd = unsafe { vault_pda.borrow_unchecked() };
    let vault = Vault::from_bytes(vd)?;
    if &ba.mint != &vault.mint { return Err(ProgramError::InvalidArgument); }

    // vault_ata sanity: owner field at byte offset 32..64 must match vault_pda; mint at 0..32
    // must match the vault's recorded mint.
    let vad = unsafe { vault_ata.borrow_unchecked() };
    if vad.len() < 64 { return Err(ProgramError::InvalidAccountData); }
    if &vad[32..64] != vault_pda.address().as_ref() { return Err(ProgramError::InvalidArgument); }
    if &vad[0..32] != &vault.mint { return Err(ProgramError::InvalidArgument); }

    SplTransfer { from: user_ata, to: vault_ata, authority: owner, amount }.invoke()?;

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.mint_to_graph(balance_ct, amount_ct, balance_ct)?;
    Ok(())
}

// ── 3: Transfer ──
//
// Atomic transfer between two BalanceAccounts. Both must already exist (call InitBalance
// first for the recipient). Single FHE graph: from -= amount, to += amount.
//
// Account layout:
//   [0] from_acct
//   [1] to_acct
//   [2] from_ct (writable)
//   [3] to_ct (writable)
//   [4] amount_ct (input)
//   [5]+ encrypt CPI accounts (payer = from_owner who signs)
//
// Data: [3, cpi_bump]
fn transfer(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [from_acct, to_acct, from_ct, to_ct, amount_ct,
         ep, cfg, dep, cpi_auth, caller, nk, from_owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !from_owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];

    let fd = unsafe { from_acct.borrow_unchecked() };
    let from = BalanceAccount::from_bytes(fd)?;
    if &from.owner != from_owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if from_ct.address().as_array() != &from.balance_ct { return Err(ProgramError::InvalidArgument); }

    let td = unsafe { to_acct.borrow_unchecked() };
    let to = BalanceAccount::from_bytes(td)?;
    if to_ct.address().as_array() != &to.balance_ct { return Err(ProgramError::InvalidArgument); }
    // Same-mint check: both balances must denominate the same SPL.
    if from.mint != to.mint { return Err(ProgramError::InvalidArgument); }

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: from_owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.transfer_graph(from_ct, to_ct, amount_ct, from_ct, to_ct)?;
    Ok(())
}

// ── 4: ViewBalance ──
//
// Decrypt-on-demand. Same pattern as PC-Token's view_balance.
//
// Account layout:
//   [0] balance_acct
//   [1] balance_ct (read-only)
//   [2] view_ct (signer, fresh keypair)
//   [3] new_authorized (caller-provided ephemeral pubkey)
//   [4]+ encrypt CPI accounts
//
// Data: [4, cpi_bump]
fn view_balance(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [ba_acct, balance_ct, view_ct, new_authorized,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];

    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ba.balance_ct { return Err(ProgramError::InvalidArgument); }

    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.copy_ciphertext(balance_ct, view_ct, new_authorized)?;
    Ok(())
}

// ── 5: TransferOut ──
//
// Rotate the user's BalanceAccount's ciphertext authority from pc-token-cold's CPI authority
// to a target program's authority, then write a Ticket binding the rotated CT to the user
// (recipient). The user's BalanceAccount is closed; rent flows back to the user.
//
// The user must trust the target program — once authority is rotated, only that program
// can run graphs against the CT.
//
// Account layout:
//   [0] balance_acct (writable; will be closed)
//   [1] balance_ct (writable; transfer_ciphertext writes here)
//   [2] target_authority (read-only; the new CPI authority pubkey, e.g.
//       PDA(["__encrypt_cpi_authority"], target_program))
//   [3] ticket_acct (PDA, will be created at seeds [b"pc_ticket", balance_ct])
//   [4] owner (signer; must match balance_acct.owner)
//   [5] rent_destination (writable; receives the closed-balance lamports — typically owner)
//   [6]+ encrypt CPI accounts
//   [last] system_program (already in encrypt CPI accounts but listed for ticket creation)
//
// Data: [5, cpi_bump, ticket_bump]
fn transfer_out(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [ba_acct, balance_ct, target_authority, ticket_acct, owner, rent_destination,
         ep, cfg, dep, cpi_auth, caller, nk, payer, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (cpi_bump, ticket_bump) = (data[0], data[1]);

    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ba.balance_ct { return Err(ProgramError::InvalidArgument); }
    let mint = ba.mint;

    // Rotate CT authority to the target program's pubkey.
    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer, event_authority: evt, system_program: sys,
        cpi_authority_bump: cpi_bump,
    };
    ctx.transfer_ciphertext(balance_ct, target_authority)?;

    // Create the Ticket PDA — bound to the recipient (= owner, the user transferring out).
    // Idempotent: same-content ticket from a prior cycle is reused.
    if ticket_acct.lamports() > 0 {
        let td = unsafe { ticket_acct.borrow_unchecked() };
        if td.len() >= Ticket::LEN {
            let existing = Ticket::from_bytes(td)?;
            if &existing.balance_ct == balance_ct.address().as_array()
                && &existing.recipient == owner.address().as_array()
                && &existing.mint == &mint
            {
                // Same ticket already exists; skip creation.
            } else {
                return Err(ProgramError::AccountAlreadyInitialized);
            }
        } else {
            return Err(ProgramError::AccountAlreadyInitialized);
        }
    } else {
        let bb = [ticket_bump];
        let seeds = [Seed::from(b"pc_ticket" as &[u8]), Seed::from(balance_ct.address().as_ref()), Seed::from(&bb)];
        CreateAccount {
            from: payer, to: ticket_acct, lamports: minimum_balance(Ticket::LEN),
            space: Ticket::LEN as u64, owner: program_id,
        }.invoke_signed(&[Signer::from(&seeds)])?;
        let td = unsafe { ticket_acct.borrow_unchecked_mut() };
        let ticket = Ticket::from_bytes_mut(td)?;
        ticket.balance_ct.copy_from_slice(balance_ct.address().as_ref());
        ticket.recipient.copy_from_slice(owner.address().as_ref());
        ticket.mint.copy_from_slice(&mint);
        ticket.bump = ticket_bump;
    }

    // Close the BalanceAccount. Rent → rent_destination.
    let lamports = ba_acct.lamports();
    ba_acct.set_lamports(0);
    rent_destination.set_lamports(rent_destination.lamports() + lamports);
    let bd2 = unsafe { ba_acct.borrow_unchecked_mut() };
    for b in bd2.iter_mut() { *b = 0; }
    Ok(())
}

// ── 6: AcceptIn ──
//
// Consume an incoming Ticket — created either by pc-token-cold's TransferOut (cold→hot
// targeting another program; this branch isn't useful for AcceptIn since rotation went
// elsewhere) or by another program's CreateInboundTicketCpi (hot→cold; the CT is now
// authorized to pc-token-cold and waiting to be claimed).
//
// Verifies the signer matches the ticket's recipient binding, then creates a BalanceAccount
// for the user referencing the now-cold-authorized CT. Closes the ticket — rent flows to
// the user.
//
// Account layout:
//   [0] ticket_acct (writable; will be closed)
//   [1] balance_acct (PDA, will be created)
//   [2] balance_ct (read; pre-existing CT, already authorized to pc-token-cold)
//   [3] mint (read)
//   [4] owner (signer; must match ticket.recipient)
//   [5] system_program (required by the inner CreateAccount CPI)
//
// Data: [6, ba_bump]
fn accept_in(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [ticket_acct, ba_acct, balance_ct, mint, owner, _sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let ba_bump = data[0];

    let td = unsafe { ticket_acct.borrow_unchecked() };
    let ticket = Ticket::from_bytes(td)?;
    if &ticket.recipient != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if &ticket.balance_ct != balance_ct.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if &ticket.mint != mint.address().as_array() { return Err(ProgramError::InvalidArgument); }

    // Create the BalanceAccount PDA pointing at the now-cold-authorized CT.
    // CRITICAL: owner must be the runtime-supplied program_id, NOT the hardcoded ID const
    // — the deployed program lives at a different address (FaikTJS...) than the placeholder
    // [8u8; 32] in source.
    let bb = [ba_bump];
    let seeds = [
        Seed::from(b"pc_balance" as &[u8]),
        Seed::from(owner.address().as_ref()),
        Seed::from(mint.address().as_ref()),
        Seed::from(&bb),
    ];
    CreateAccount {
        from: owner, to: ba_acct, lamports: minimum_balance(BalanceAccount::LEN),
        space: BalanceAccount::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let bd = unsafe { ba_acct.borrow_unchecked_mut() };
    let ba = BalanceAccount::from_bytes_mut(bd)?;
    ba.owner.copy_from_slice(owner.address().as_ref());
    ba.mint.copy_from_slice(mint.address().as_ref());
    ba.balance_ct.copy_from_slice(balance_ct.address().as_ref());
    ba.bump = ba_bump;

    // Close the ticket — rent → owner.
    let lamports = ticket_acct.lamports();
    ticket_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + lamports);
    let td2 = unsafe { ticket_acct.borrow_unchecked_mut() };
    for b in td2.iter_mut() { *b = 0; }
    Ok(())
}

// ── 7: CreateInboundTicketCpi ──
//
// Permissionless helper. Other programs CPI in to write a Ticket bound to a user, after
// they've done their own `transfer_ciphertext` rotation pointing the CT at pc-token-cold's
// CPI authority.
//
// Permissionless because the security comes from two facts:
//   (a) Only the user matching ticket.recipient can claim the CT via AcceptIn.
//   (b) The CT must already be authorized to pc-token-cold for AcceptIn to be useful;
//       only the prior CT authority could have rotated it there.
//
// Spam tickets are possible but cost the spammer rent and don't grant access to anyone's
// balance.
//
// Account layout:
//   [0] ticket_acct (PDA, will be created)
//   [1] balance_ct (read; the CT being transferred — must already be rotated to pc-token-cold)
//   [2] recipient (read; the user this ticket is bound to)
//   [3] mint (read)
//   [4] payer (signer; pays rent for the ticket account)
//   [5] system_program
//
// Data: [7, ticket_bump]
fn create_inbound_ticket_cpi(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [ticket_acct, balance_ct, recipient, mint, payer, _sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let ticket_bump = data[0];

    // Idempotent: if a ticket already exists at this PDA with matching contents (same
    // balance_ct + recipient + mint), succeed as a no-op. Same balance_ct can recur across
    // deposit↔withdraw cycles and merge_* paths don't close the ticket; this handles that
    // accumulation without leaving stranded PDAs.
    if ticket_acct.lamports() > 0 {
        let td = unsafe { ticket_acct.borrow_unchecked() };
        if td.len() >= Ticket::LEN {
            let existing = Ticket::from_bytes(td)?;
            if &existing.balance_ct == balance_ct.address().as_array()
                && &existing.recipient == recipient.address().as_array()
                && &existing.mint == mint.address().as_array()
            {
                return Ok(());
            }
        }
        return Err(ProgramError::AccountAlreadyInitialized);
    }

    let bb = [ticket_bump];
    let seeds = [Seed::from(b"pc_ticket" as &[u8]), Seed::from(balance_ct.address().as_ref()), Seed::from(&bb)];
    CreateAccount {
        from: payer, to: ticket_acct, lamports: minimum_balance(Ticket::LEN),
        space: Ticket::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;
    let td = unsafe { ticket_acct.borrow_unchecked_mut() };
    let ticket = Ticket::from_bytes_mut(td)?;
    ticket.balance_ct.copy_from_slice(balance_ct.address().as_ref());
    ticket.recipient.copy_from_slice(recipient.address().as_ref());
    ticket.mint.copy_from_slice(mint.address().as_ref());
    ticket.bump = ticket_bump;
    Ok(())
}

// ── 8: MergeIn ──
//
// Variant of accept_in for the case where the user already has a cold BalanceAccount for
// this mint. Runs `mint_to_graph(existing, incoming, existing)` to add the rotated incoming
// CT into the existing cold balance. The incoming CT is left orphaned (rent stays).
//
// Account layout:
//   [0]  ticket_acct (read; verifies provenance)
//   [1]  ba_acct (read; verifies signer + mint)
//   [2]  existing_balance_ct (writable; receives the merge)
//   [3]  incoming_balance_ct (writable; the rotated CT — read by graph)
//   [4]  mint (read)
//   [5]  owner (signer; must match ticket.recipient and ba.owner)
//   [6]+ encrypt CPI accounts
//
// Data: [8, cpi_bump]
fn merge_in(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [ticket_acct, ba_acct, existing_balance_ct, incoming_balance_ct, mint, owner,
         ep, cfg, dep, cpi_auth, caller, nk, payer, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if !payer.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];

    // Verify ticket bindings.
    let td = unsafe { ticket_acct.borrow_unchecked() };
    let ticket = Ticket::from_bytes(td)?;
    if &ticket.recipient != owner.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }
    if &ticket.balance_ct != incoming_balance_ct.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }
    if &ticket.mint != mint.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }

    // Verify existing BalanceAccount.
    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }
    if &ba.mint != mint.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }
    if existing_balance_ct.address().as_array() != &ba.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    // Run merge graph: existing += incoming. Both CTs are pc-token-cold-authorized
    // (existing always was, incoming was rotated to us by the source program's withdraw).
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
// First step of cold → SPL unwrap. Burns `amount` from the user's encrypted cold balance
// (FHE conditional: only if balance >= amount; no-op otherwise). Creates a WithdrawalReceipt
// recording the requested amount; the actual burned amount lands in `burned_ct`.
//
// Account layout:
//   [0]  vault_pda (read; verifies SPL mint)
//   [1]  ba_acct (read; verifies signer + mint)
//   [2]  receipt_acct (PDA, will be created at seeds [b"pc_receipt", burned_ct])
//   [3]  balance_ct (writable; FHE: -= amount on success)
//   [4]  amount_ct (read; user-encrypted EUint128 input)
//   [5]  burned_ct (writable; pre-allocated CT, FHE writes the actual burned amount)
//   [6]  mint (read; SPL mint identifier)
//   [7]+ encrypt CPI accounts
//
// Data: [9, receipt_bump, cpi_bump, amount(u64 LE)]
fn unwrap_burn(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [vault_pda, ba_acct, receipt_acct, balance_ct, amount_ct, burned_ct, mint,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 10 { return Err(ProgramError::InvalidInstructionData); }
    let (rb, cb) = (data[0], data[1]);
    let amount = u64::from_le_bytes(data[2..10].try_into().unwrap());
    if amount == 0 { return Err(ProgramError::InvalidArgument); }

    // Verify vault's mint matches the requested mint.
    let vd = unsafe { vault_pda.borrow_unchecked() };
    let vault = Vault::from_bytes(vd)?;
    if &vault.mint != mint.address().as_array() { return Err(ProgramError::InvalidArgument); }

    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if &ba.mint != mint.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ba.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    // Create the receipt PDA.
    let bb = [rb];
    let seeds = [Seed::from(b"pc_receipt" as &[u8]), Seed::from(burned_ct.address().as_ref()), Seed::from(&bb)];
    CreateAccount {
        from: owner, to: receipt_acct, lamports: minimum_balance(WithdrawalReceipt::LEN),
        space: WithdrawalReceipt::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    // Run the FHE burn graph. burned_ct is pre-created via gRPC (value=0); graph writes
    // either `amount` (if sufficient) or 0.
    let ctx = EncryptContext {
        encrypt_program: ep, config: cfg, deposit: dep,
        cpi_authority: cpi_auth, caller_program: caller,
        network_encryption_key: nk, payer: owner, event_authority: evt, system_program: sys,
        cpi_authority_bump: cb,
    };
    ctx.unwrap_burn_graph(balance_ct, amount_ct, balance_ct, burned_ct)?;

    // Populate the receipt.
    let rd = unsafe { receipt_acct.borrow_unchecked_mut() };
    let r = WithdrawalReceipt::from_bytes_mut(rd)?;
    r.owner.copy_from_slice(owner.address().as_ref());
    r.amount = amount.to_le_bytes();
    r.pending_digest = [0u8; 32];
    r.mint.copy_from_slice(mint.address().as_ref());
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
// Final step. Reads decrypted burned amount, verifies digest, releases SPL (if burn matched
// the requested amount), then closes EVERY satellite account that's now dead: the receipt,
// the burned_ct (FHE input now consumed), and the decryption request. All rent → owner.
//
// If the FHE burn no-op'd (insufficient balance, burned=0), the SPL release is skipped but
// the closes still run — user reclaims rent for receipt + burned_ct + request_acct, keeps
// their encrypted balance unchanged.
//
// Account layout:
//   [0]  receipt_acct (writable; will be closed)
//   [1]  vault_pda (read; SPL transfer authority signer)
//   [2]  vault_ata (writable; SPL source)
//   [3]  user_ata (writable; SPL destination)
//   [4]  mint (read)
//   [5]  request_acct (writable; closed via CPI close_decryption_request)
//   [6]  burned_ct (writable; closed via CPI close_ciphertext)
//   [7]  owner (signer; rent destination)
//   [8]  spl_token_program
//   [9..] encrypt CPI accounts: ep, cfg, dep, cpi_auth, caller, nk, evt, sys
//
// Data: [11, cpi_bump]
fn unwrap_complete(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [receipt_acct, vault_pda, vault_ata, user_ata, mint, request_acct, burned_ct,
         owner, _spl,
         ep, cfg, dep, cpi_auth, caller, nk, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];

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
        let vd = unsafe { vault_pda.borrow_unchecked() };
        let vault = Vault::from_bytes(vd)?;
        if &vault.mint != mint.address().as_array() {
            return Err(ProgramError::InvalidArgument);
        }
        let vad = unsafe { vault_ata.borrow_unchecked() };
        if vad.len() < 64 { return Err(ProgramError::InvalidAccountData); }
        if &vad[32..64] != vault_pda.address().as_ref() {
            return Err(ProgramError::InvalidArgument);
        }
        let vbb = [vault.bump];
        let vs = [Seed::from(b"pc_vault" as &[u8]), Seed::from(mint.address().as_ref()), Seed::from(&vbb)];
        SplTransfer { from: vault_ata, to: user_ata, authority: vault_pda, amount: requested }
            .invoke_signed(&[Signer::from(&vs)])?;
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

// ── 12: ReleaseSpl ──
//
// Settlement-program escape hatch. A whitelisted minter program CPIs in to release SPL
// backing on behalf of an FHE-burn it performed against one of its own encrypted balances.
// cold does no FHE accounting here — it just acts as the SPL custodian and trusts the
// caller to have correctly debited its own books before invoking.
//
// Authorization:
//   - The caller signs a PDA at seeds `[b"cold_release"]` under its program ID (via
//     `invoke_signed`). The signer's address must be either:
//       a) the bootstrap constant `TRUSTED_DAGON_POOL_RELEASE_AUTHORITY` (legacy), OR
//       b) present in the optional `MintersRegistry` account passed as a trailing account
//          (modern; lets us add minters without redeploying cold).
//   - When a registry account is passed, both checks are tried (logical OR). When it isn't,
//     only the constant is honored. This is soft-migration: pre-init flows still work.
//
// Account layout:
//   [0]  vault_pda (read; cold's vault PDA at [b"pc_vault", mint])
//   [1]  vault_ata (writable; SPL source — owned by vault_pda)
//   [2]  user_ata (writable; SPL destination)
//   [3]  mint (read)
//   [4]  caller_release_auth (signer; PDA of a trusted minter)
//   [5]  spl_token_program
//   [6]  (optional) minters_registry (read; PDA at [b"cold_minters"])
//
// Data: [12, amount(u64 LE)]
fn release_spl(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [vault_pda, vault_ata, user_ata, mint, caller_release_auth, _spl, rest @ ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !caller_release_auth.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if data.len() < 8 { return Err(ProgramError::InvalidInstructionData); }
    let amount = u64::from_le_bytes(data[..8].try_into().unwrap());
    if amount == 0 { return Err(ProgramError::InvalidArgument); }

    let signer_addr = caller_release_auth.address().as_array();
    let mut authorized = signer_addr == &TRUSTED_DAGON_POOL_RELEASE_AUTHORITY;
    if !authorized {
        if let [registry_acct, ..] = rest {
            let rd = unsafe { registry_acct.borrow_unchecked() };
            let registry = MintersRegistry::from_bytes(rd)?;
            if registry.contains(signer_addr) {
                authorized = true;
            }
        }
    }
    if !authorized {
        return Err(ProgramError::InvalidArgument);
    }

    let vd = unsafe { vault_pda.borrow_unchecked() };
    let vault = Vault::from_bytes(vd)?;
    if &vault.mint != mint.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }
    let vad = unsafe { vault_ata.borrow_unchecked() };
    if vad.len() < 64 { return Err(ProgramError::InvalidAccountData); }
    if &vad[32..64] != vault_pda.address().as_ref() {
        return Err(ProgramError::InvalidArgument);
    }
    if &vad[0..32] != &vault.mint {
        return Err(ProgramError::InvalidArgument);
    }

    let vbb = [vault.bump];
    let vs = [Seed::from(b"pc_vault" as &[u8]), Seed::from(mint.address().as_ref()), Seed::from(&vbb)];
    SplTransfer { from: vault_ata, to: user_ata, authority: vault_pda, amount }
        .invoke_signed(&[Signer::from(&vs)])?;
    Ok(())
}

// ── 13: CloseBurn ──
//
// First step of cold balance teardown. Drains the encrypted balance to 0 and writes the
// pre-burn amount into `burned_ct`. Same shape as `unwrap_burn` but uses close_burn_graph
// (no sufficiency check — always burns everything). Creates a receipt PDA the user can
// later finalize with `close_complete`.
//
// Account layout:
//   [0]  vault_pda (read; verifies SPL mint)
//   [1]  ba_acct (read; verifies signer + mint)
//   [2]  receipt_acct (PDA, will be created at seeds [b"pc_receipt", burned_ct])
//   [3]  balance_ct (writable; FHE: -> 0)
//   [4]  burned_ct (writable; pre-allocated CT, FHE writes the pre-burn balance here)
//   [5]  mint (read)
//   [6]+ encrypt CPI accounts
//
// Data: [13, receipt_bump, cpi_bump]
fn close_burn(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [vault_pda, ba_acct, receipt_acct, balance_ct, burned_ct, mint,
         ep, cfg, dep, cpi_auth, caller, nk, owner, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 2 { return Err(ProgramError::InvalidInstructionData); }
    let (rb, cb) = (data[0], data[1]);

    let vd = unsafe { vault_pda.borrow_unchecked() };
    let vault = Vault::from_bytes(vd)?;
    if &vault.mint != mint.address().as_array() { return Err(ProgramError::InvalidArgument); }

    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if &ba.mint != mint.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ba.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let bb = [rb];
    let seeds = [Seed::from(b"pc_receipt" as &[u8]), Seed::from(burned_ct.address().as_ref()), Seed::from(&bb)];
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
    // amount field is unused for close (release amount comes entirely from the decrypted
    // burned_ct); leave at 0 for clarity. close_complete reads the decrypted plaintext,
    // not the receipt amount.
    r.amount = 0u64.to_le_bytes();
    r.pending_digest = [0u8; 32];
    r.mint.copy_from_slice(mint.address().as_ref());
    r.bump = rb;
    Ok(())
}

// ── 14: CloseComplete ──
//
// Final step of cold balance teardown. Verifies the digest, releases SPL of `burned`
// (the decrypted pre-burn balance) to the user, then closes EVERY satellite account that
// was carrying rent: the BA itself, the now-zero balance_ct, the burned_ct, the
// decryption request, and the receipt. All rent → owner.
//
// If burned == 0 (BA was empty), the SPL transfer is skipped but every close still runs.
//
// Account layout:
//   [0]  receipt_acct (writable; will be closed last)
//   [1]  vault_pda (read)
//   [2]  vault_ata (writable; SPL source)
//   [3]  user_ata (writable; SPL destination)
//   [4]  mint (read)
//   [5]  request_acct (writable; will be closed via CPI close_decryption_request)
//   [6]  ba_acct (writable; will be zeroed locally)
//   [7]  balance_ct (writable; will be closed via CPI close_ciphertext)
//   [8]  burned_ct (writable; will be closed via CPI close_ciphertext)
//   [9]  owner (signer; rent destination for everything)
//   [10] spl_token_program
//   [11..] encrypt CPI accounts: ep, cfg, dep, cpi_auth, caller, nk, evt, sys
//
// Data: [14, cpi_bump]
fn close_complete(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [receipt_acct, vault_pda, vault_ata, user_ata, mint, request_acct,
         ba_acct, balance_ct, burned_ct, owner, _spl,
         ep, cfg, dep, cpi_auth, caller, nk, evt, sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !owner.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let cpi_bump = data[0];

    let rd = unsafe { receipt_acct.borrow_unchecked() };
    let r = WithdrawalReceipt::from_bytes(rd)?;
    if owner.address().as_array() != &r.owner { return Err(ProgramError::InvalidArgument); }
    if mint.address().as_array() != &r.mint { return Err(ProgramError::InvalidArgument); }
    let digest = r.pending_digest;

    let bd = unsafe { ba_acct.borrow_unchecked() };
    let ba = BalanceAccount::from_bytes(bd)?;
    if &ba.owner != owner.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if &ba.mint != mint.address().as_array() { return Err(ProgramError::InvalidArgument); }
    if balance_ct.address().as_array() != &ba.balance_ct {
        return Err(ProgramError::InvalidArgument);
    }

    let req_data = unsafe { request_acct.borrow_unchecked() };
    let burned: &u128 = accounts::read_decrypted_verified::<Uint128>(req_data, &digest)?;
    let release_amount = *burned as u64;

    if release_amount > 0 {
        let vd = unsafe { vault_pda.borrow_unchecked() };
        let vault = Vault::from_bytes(vd)?;
        if &vault.mint != mint.address().as_array() {
            return Err(ProgramError::InvalidArgument);
        }
        let vad = unsafe { vault_ata.borrow_unchecked() };
        if vad.len() < 64 { return Err(ProgramError::InvalidAccountData); }
        if &vad[32..64] != vault_pda.address().as_ref() {
            return Err(ProgramError::InvalidArgument);
        }
        let vbb = [vault.bump];
        let vs = [Seed::from(b"pc_vault" as &[u8]), Seed::from(mint.address().as_ref()), Seed::from(&vbb)];
        SplTransfer { from: vault_ata, to: user_ata, authority: vault_pda, amount: release_amount }
            .invoke_signed(&[Signer::from(&vs)])?;
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

    let bal = ba_acct.lamports();
    ba_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + bal);
    let bd2 = unsafe { ba_acct.borrow_unchecked_mut() };
    for b in bd2.iter_mut() { *b = 0; }

    let rl = receipt_acct.lamports();
    receipt_acct.set_lamports(0);
    owner.set_lamports(owner.lamports() + rl);
    let rd2 = unsafe { receipt_acct.borrow_unchecked_mut() };
    for b in rd2.iter_mut() { *b = 0; }
    Ok(())
}

// ── 15: InitMintersRegistry ──
//
// One-shot create of the singleton trusted-minter registry at PDA `[b"cold_minters"]`.
// The signer becomes the registry admin (only address that can later add/remove minters
// via `set_trusted_minter`). The list is empty at init; admin populates it after.
//
// Account layout:
//   [0]  registry_acct (PDA, will be created at [b"cold_minters"])
//   [1]  admin (signer; written into registry.admin)
//   [2]  system_program
//
// Data: [15, registry_bump]
fn init_minters_registry(
    program_id: &Address, accounts: &[AccountView], data: &[u8],
) -> ProgramResult {
    let [registry_acct, admin, _sys, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !admin.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.is_empty() { return Err(ProgramError::InvalidInstructionData); }
    let bump = data[0];

    let bb = [bump];
    let seeds = [Seed::from(b"cold_minters" as &[u8]), Seed::from(&bb)];
    CreateAccount {
        from: admin, to: registry_acct, lamports: minimum_balance(MintersRegistry::LEN),
        space: MintersRegistry::LEN as u64, owner: program_id,
    }.invoke_signed(&[Signer::from(&seeds)])?;

    let rd = unsafe { registry_acct.borrow_unchecked_mut() };
    let r = MintersRegistry::from_bytes_mut(rd)?;
    r.admin.copy_from_slice(admin.address().as_ref());
    r.bump = bump;
    r.len = 0;
    r._pad = [0u8; 6];
    for slot in r.authorities.iter_mut() {
        *slot = [0u8; 32];
    }
    Ok(())
}

// ── 16: SetTrustedMinter ──
//
// Admin-only add/remove of a release-authority pubkey from the registry. Idempotent on
// add (already-present is a no-op success) and on remove (not-present is a no-op success).
//
// Account layout:
//   [0]  registry_acct (writable; PDA at [b"cold_minters"])
//   [1]  admin (signer; must match registry.admin)
//
// Data: [16, op, authority(32 bytes)]
//   op = 0  → add
//   op = 1  → remove
fn set_trusted_minter(accounts: &[AccountView], data: &[u8]) -> ProgramResult {
    let [registry_acct, admin, ..] = accounts
    else { return Err(ProgramError::NotEnoughAccountKeys); };
    if !admin.is_signer() { return Err(ProgramError::MissingRequiredSignature); }
    if data.len() < 33 { return Err(ProgramError::InvalidInstructionData); }
    let op = data[0];
    let mut authority = [0u8; 32];
    authority.copy_from_slice(&data[1..33]);

    let rd = unsafe { registry_acct.borrow_unchecked_mut() };
    let r = MintersRegistry::from_bytes_mut(rd)?;
    if &r.admin != admin.address().as_array() {
        return Err(ProgramError::InvalidArgument);
    }

    match op {
        0 => {
            if r.contains(&authority) { return Ok(()); }
            if (r.len as usize) >= MAX_TRUSTED_MINTERS {
                return Err(ProgramError::InvalidArgument);
            }
            let idx = r.len as usize;
            r.authorities[idx] = authority;
            r.len += 1;
        }
        1 => {
            let idx = match r.position(&authority) {
                Some(i) => i,
                None => return Ok(()),
            };
            let last = (r.len as usize) - 1;
            if idx != last {
                r.authorities[idx] = r.authorities[last];
            }
            r.authorities[last] = [0u8; 32];
            r.len -= 1;
        }
        _ => return Err(ProgramError::InvalidArgument),
    }
    Ok(())
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use encrypt_types::graph::{get_node, parse_graph, GraphNodeKind};
    use encrypt_types::identifier::*;
    use encrypt_types::types::FheType;
    use super::{mint_to_graph, transfer_graph};

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

    #[test] fn transfer_ok() {
        let r = run_mock(transfer_graph, &[1000, 500, 300], &[T, T, T]);
        assert_eq!(r[0], 700, "from -= 300");
        assert_eq!(r[1], 800, "to += 300");
    }
    #[test] fn transfer_insufficient() {
        let r = run_mock(transfer_graph, &[100, 500, 300], &[T, T, T]);
        assert_eq!(r[0], 100, "from unchanged");
        assert_eq!(r[1], 500, "to unchanged");
    }
    #[test] fn transfer_exact() {
        // transferring entire balance should zero source, fully credit destination.
        let r = run_mock(transfer_graph, &[300, 500, 300], &[T, T, T]);
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 800);
    }

    #[test] fn graph_shapes() {
        swap_shapes(mint_to_graph, 2, 1);
        swap_shapes(transfer_graph, 3, 2);
    }
    fn swap_shapes(f: fn() -> Vec<u8>, ni: u8, no: u8) {
        let g = f(); let pg = parse_graph(&g).unwrap();
        assert_eq!(pg.header().num_inputs(), ni);
        assert_eq!(pg.header().num_outputs(), no);
    }
}
