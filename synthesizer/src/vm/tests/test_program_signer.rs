// Copyright (c) 2019-2026 Provable Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! End-to-end tests for program-owned signers (exploratory prototype, Mechanism 1).
//!
//! A `vault.aleo` program owns `Y = X + Ht(signer, vault.aleo, X)·G`. Executions rooted in
//! `vault.aleo` use `Y` as their signer, with no signature: the vault's code checks a 2-of-3
//! approval with `sign.verify`, then spends a `credits.aleo` record owned by `Y`.

use super::*;

use console::{
    account::{Address, Signature, ViewKey},
    network::ConsensusVersion,
    program::{
        Ciphertext,
        Identifier,
        Literal,
        PROGRAM_SIGNER_KIND_SIGNER,
        Plaintext,
        ProgramID,
        ProgramSigner,
        Record,
        RequestSigner,
        Value,
        program_signer_tweak_domain,
    },
    types::{Scalar, U64},
};
use snarkvm_ledger_block::{Block, Transaction};
use snarkvm_synthesizer_program::Program;
use snarkvm_utilities::TestRng;

/// Returns the vault program with the given name.
fn vault_program(name: &str) -> Program<CurrentNetwork> {
    let domain = program_signer_tweak_domain::<CurrentNetwork>();
    let kind = PROGRAM_SIGNER_KIND_SIGNER;
    Program::from_str(&format!(
        r"
import credits.aleo;

program {name}.aleo;

struct approval:
    vault as address;
    nonce as u64;
    recipient as address;
    amount as u64;

record config:
    owner as address.private;
    signer0 as address.private;
    signer1 as address.private;
    signer2 as address.private;
    threshold as u8.private;
    nonce as u64.private;

mapping used:
    key as field.public;
    value as boolean.public;

// Creates the vault's config. Only the holder of the internal secret `x` can create it, and only once.
function create:
    input r0 as scalar.private;
    input r1 as address.private;
    input r2 as address.private;
    input r3 as address.private;
    input r4 as u8.private;
    assert.eq self.caller self.signer;
    mul aleo::GENERATOR r0 into r5;
    cast r5 into r6 as group.x;
    cast r5 into r7 as group.y;
    cast {name}.aleo into r8 as field;
    cast {domain} {kind}field r8 r6 r7 into r9 as [field; 5u32];
    hash.psd4 r9 into r10 as scalar;
    mul aleo::GENERATOR r10 into r11;
    add r5 r11 into r12;
    cast self.signer into r13 as group;
    assert.eq r12 r13;
    cast r6 r7 into r14 as [field; 2u32];
    hash.psd2 r14 into r15 as field;
    cast self.signer r1 r2 r3 r4 0u64 into r16 as config.record;
    async create r15 into r17;
    output r16 as config.record;
    output r17 as {name}.aleo/create.future;

finalize create:
    input r0 as field.public;
    contains used[r0] into r1;
    assert.eq r1 false;
    set true into used[r0];

// Transfers `amount` of a vault-owned credits record to `recipient`, given k-of-3 approvals.
function execute_transfer:
    input r0 as config.record;
    input r1 as credits.aleo/credits.record;
    input r2 as address.private;
    input r3 as u64.private;
    input r4 as signature.private;
    input r5 as signature.private;
    input r6 as signature.private;
    assert.eq self.caller self.signer;
    cast self.signer r0.nonce r2 r3 into r7 as approval;
    sign.verify r4 r0.signer0 r7 into r8;
    sign.verify r5 r0.signer1 r7 into r9;
    sign.verify r6 r0.signer2 r7 into r10;
    ternary r8 1u8 0u8 into r11;
    ternary r9 1u8 0u8 into r12;
    ternary r10 1u8 0u8 into r13;
    add r11 r12 into r14;
    add r14 r13 into r15;
    gte r15 r0.threshold into r16;
    assert.eq r16 true;
    call credits.aleo/transfer_private r1 r2 r3 into r17 r18;
    add r0.nonce 1u64 into r19;
    cast r0.owner r0.signer0 r0.signer1 r0.signer2 r0.threshold r19 into r20 as config.record;
    output r20 as config.record;
    output r17 as credits.aleo/credits.record;
    output r18 as credits.aleo/credits.record;

constructor:
    assert.eq edition 0u16;
"
    ))
    .unwrap()
}

/// Builds a transaction for the given authorization, with a public fee paid by `fee_payer`.
fn execute_with_fee(
    vm: &VM<CurrentNetwork, LedgerType>,
    fee_payer: &PrivateKey<CurrentNetwork>,
    authorization: Authorization<CurrentNetwork>,
    rng: &mut TestRng,
) -> Transaction<CurrentNetwork> {
    let execution_id = authorization.to_execution_id().unwrap();
    let fee = vm.authorize_fee_public(fee_payer, 1_000_000, 0, execution_id, rng).unwrap();
    let transaction = vm.execute_authorization(authorization, Some(fee), None, rng).unwrap();
    vm.check_transaction(&transaction, None, rng).unwrap();
    transaction
}

/// Adds a block with the given transactions, and returns it.
fn add_block(
    vm: &VM<CurrentNetwork, LedgerType>,
    genesis: &PrivateKey<CurrentNetwork>,
    transactions: &[Transaction<CurrentNetwork>],
    rng: &mut TestRng,
) -> Block<CurrentNetwork> {
    let block = sample_next_block(vm, genesis, transactions, rng).unwrap();
    vm.add_next_block(&block).unwrap();
    block
}

/// Asserts that every transaction in the block was accepted.
fn assert_all_accepted(block: &Block<CurrentNetwork>, expected: usize) {
    assert_eq!(block.aborted_transaction_ids().len(), 0, "aborted transactions");
    assert_eq!(block.transactions().num_accepted(), expected, "accepted transactions");
    assert_eq!(block.transactions().num_rejected(), 0, "rejected transactions");
}

/// Returns the records in the transaction that the view key owns, decrypted.
fn owned_records(
    transaction: &Transaction<CurrentNetwork>,
    view_key: &ViewKey<CurrentNetwork>,
) -> Vec<Record<CurrentNetwork, Plaintext<CurrentNetwork>>> {
    transaction
        .records()
        .filter(|(_, record): &(_, &Record<CurrentNetwork, Ciphertext<CurrentNetwork>>)| record.is_owner(view_key))
        .map(|(_, record)| record.decrypt(view_key).unwrap())
        .collect()
}

/// Returns the microcredits in a credits record.
fn microcredits(record: &Record<CurrentNetwork, Plaintext<CurrentNetwork>>) -> Option<u64> {
    match record.data().get(&Identifier::from_str("microcredits").unwrap()) {
        Some(console::program::Entry::Private(Plaintext::Literal(Literal::U64(amount), _))) => Some(**amount),
        _ => None,
    }
}

/// Returns a signature by `private_key` over the vault approval.
fn approve(
    private_key: &PrivateKey<CurrentNetwork>,
    vault: &Address<CurrentNetwork>,
    nonce: u64,
    recipient: &Address<CurrentNetwork>,
    amount: u64,
    rng: &mut TestRng,
) -> Value<CurrentNetwork> {
    let message = Plaintext::<CurrentNetwork>::from_str(&format!(
        "{{ vault: {vault}, nonce: {nonce}u64, recipient: {recipient}, amount: {amount}u64 }}"
    ))
    .unwrap();
    let signature = Signature::sign(private_key, &message.to_fields().unwrap(), rng).unwrap();
    Value::from(Literal::Signature(Box::new(signature)))
}

#[test]
fn test_program_signer_vault_end_to_end() {
    let rng = &mut TestRng::default();
    let genesis = sample_genesis_private_key(rng);
    let vm = sample_vm_at_height(CurrentNetwork::CONSENSUS_HEIGHT(ConsensusVersion::latest()).unwrap(), rng);

    // Ordinary accounts: Alice funds the vault, Bob receives from it, and three signers approve.
    let alice = PrivateKey::<CurrentNetwork>::new(rng).unwrap();
    let alice_view_key = ViewKey::try_from(&alice).unwrap();
    let alice_address = Address::try_from(&alice).unwrap();
    let bob = PrivateKey::<CurrentNetwork>::new(rng).unwrap();
    let bob_view_key = ViewKey::try_from(&bob).unwrap();
    let bob_address = Address::try_from(&bob).unwrap();
    let signers: Vec<_> = (0..3).map(|_| PrivateKey::<CurrentNetwork>::new(rng).unwrap()).collect();
    let signer_addresses: Vec<_> = signers.iter().map(|key| Address::try_from(key).unwrap()).collect();

    // The vault: a program-owned address `Y`, with internal secret `x` and view key `y`.
    let vault_id = ProgramID::<CurrentNetwork>::from_str("vault.aleo").unwrap();
    let x = Scalar::<CurrentNetwork>::rand(rng);
    let vault = ProgramSigner::from_internal_secret(vault_id, x).unwrap();
    let vault_address = *vault.address();
    let vault_view_key = vault.view_key();
    let vault_signer = RequestSigner::Program(vault);
    println!("vault address Y = {vault_address}");

    // Deploy `vault.aleo`.
    let deployment = vm.deploy(&genesis, &vault_program("vault"), None, 0, None, rng).unwrap();
    assert_all_accepted(&add_block(&vm, &genesis, &[deployment], rng), 1);

    // Fund Alice with a private record (an ordinary execution).
    let inputs = [Value::from(Literal::Address(alice_address)), Value::from(Literal::U64(U64::new(10_000_000)))];
    let transaction = vm
        .execute(&genesis, ("credits.aleo", "transfer_public_to_private"), inputs.iter(), None, 0, None, rng)
        .unwrap();
    assert_all_accepted(&add_block(&vm, &genesis, std::slice::from_ref(&transaction), rng), 1);
    let alice_record = owned_records(&transaction, &alice_view_key).pop().unwrap();

    // Alice funds the vault with `transfer_private` to `Y` (an ordinary execution).
    let inputs = [
        Value::Record(alice_record),
        Value::from(Literal::Address(vault_address)),
        Value::from(Literal::U64(U64::new(3_000_000))),
    ];
    let authorization = vm.authorize(&alice, "credits.aleo", "transfer_private", inputs, rng).unwrap();
    let transaction = execute_with_fee(&vm, &genesis, authorization, rng);
    assert_all_accepted(&add_block(&vm, &genesis, std::slice::from_ref(&transaction), rng), 1);
    // The record sent to `Y` decrypts with `y`.
    let vault_credits = owned_records(&transaction, &vault_view_key).pop().unwrap();
    assert_eq!(microcredits(&vault_credits), Some(3_000_000));
    assert_eq!(**vault_credits.owner(), vault_address);

    // The vault creates its config, as itself (keyless, rooted in `vault.aleo`).
    let create_inputs = [
        Value::from(Literal::Scalar(x)),
        Value::from(Literal::Address(signer_addresses[0])),
        Value::from(Literal::Address(signer_addresses[1])),
        Value::from(Literal::Address(signer_addresses[2])),
        Value::from_str("2u8").unwrap(),
    ];
    let authorization =
        vm.authorize_with_signer(&vault_signer, "vault.aleo", "create", create_inputs.clone(), rng).unwrap();
    let transaction = execute_with_fee(&vm, &genesis, authorization, rng);
    // The on-chain transaction format is unchanged: it round-trips through bytes.
    assert_eq!(transaction, Transaction::read_le(&transaction.to_bytes_le().unwrap()[..]).unwrap());
    assert_all_accepted(&add_block(&vm, &genesis, std::slice::from_ref(&transaction), rng), 1);
    let config = owned_records(&transaction, &vault_view_key).pop().unwrap();
    assert_eq!(**config.owner(), vault_address);

    // A second `create` for the same `X` is rejected by the vault's registry.
    let authorization = vm.authorize_with_signer(&vault_signer, "vault.aleo", "create", create_inputs, rng).unwrap();
    let transaction = execute_with_fee(&vm, &genesis, authorization, rng);
    let block = add_block(&vm, &genesis, std::slice::from_ref(&transaction), rng);
    assert_eq!(block.transactions().num_rejected(), 1, "the second create must be rejected");

    // With only one valid approval, the vault's own code refuses to authorize the transfer.
    let amount = 1_000_000u64;
    let bogus = PrivateKey::<CurrentNetwork>::new(rng).unwrap();
    let one_approval = [
        Value::Record(config.clone()),
        Value::Record(vault_credits.clone()),
        Value::from(Literal::Address(bob_address)),
        Value::from(Literal::U64(U64::new(amount))),
        approve(&signers[0], &vault_address, 0, &bob_address, amount, rng),
        approve(&bogus, &vault_address, 0, &bob_address, amount, rng),
        approve(&signers[2], &vault_address, 0, &bob_address, amount + 1, rng),
    ];
    // Note: `Authorize` mode does not check circuit satisfiability, so the failure surfaces when proving.
    let authorization =
        vm.authorize_with_signer(&vault_signer, "vault.aleo", "execute_transfer", one_approval, rng).unwrap();
    let execution_id = authorization.to_execution_id().unwrap();
    let fee = vm.authorize_fee_public(&genesis, 1_000_000, 0, execution_id, rng).unwrap();
    let result = vm.execute_authorization(authorization, Some(fee), None, rng);
    let error = result.expect_err("a transfer with one approval must fail");
    println!("one approval: {error}");

    // A wrong view key `y` cannot act as the vault.
    let wrong_y = ProgramSigner::new_unchecked(
        vault_id,
        PROGRAM_SIGNER_KIND_SIGNER,
        *vault.internal(),
        Scalar::<CurrentNetwork>::rand(rng),
    )
    .unwrap();
    let two_approvals = [
        Value::Record(config.clone()),
        Value::Record(vault_credits.clone()),
        Value::from(Literal::Address(bob_address)),
        Value::from(Literal::U64(U64::new(amount))),
        approve(&signers[0], &vault_address, 0, &bob_address, amount, rng),
        approve(&bogus, &vault_address, 0, &bob_address, amount, rng),
        approve(&signers[2], &vault_address, 0, &bob_address, amount, rng),
    ];
    let result = vm.authorize_with_signer(
        &RequestSigner::Program(wrong_y),
        "vault.aleo",
        "execute_transfer",
        two_approvals.clone(),
        rng,
    );
    println!("wrong y: {}", result.as_ref().err().map(|e| e.to_string()).unwrap_or_default());
    assert!(result.is_err(), "a wrong y must fail");

    // `Y` cannot be the signer of an execution rooted in another program.
    let inputs = [
        Value::Record(vault_credits.clone()),
        Value::from(Literal::Address(bob_address)),
        Value::from(Literal::U64(U64::new(amount))),
    ];
    let result = vm.authorize_with_signer(&vault_signer, "credits.aleo", "transfer_private", inputs, rng);
    println!("foreign root: {}", result.as_ref().err().map(|e| e.to_string()).unwrap_or_default());
    assert!(result.is_err(), "a program-signer execution rooted in credits.aleo must fail");

    // With 2-of-3 approvals, the vault transfers to Bob, as itself.
    let authorization =
        vm.authorize_with_signer(&vault_signer, "vault.aleo", "execute_transfer", two_approvals, rng).unwrap();
    assert_eq!(authorization.len(), 2, "vault.aleo/execute_transfer and credits.aleo/transfer_private");
    let transaction = execute_with_fee(&vm, &genesis, authorization, rng);
    assert_all_accepted(&add_block(&vm, &genesis, std::slice::from_ref(&transaction), rng), 1);

    // Bob receives the amount.
    let bob_records = owned_records(&transaction, &bob_view_key);
    assert_eq!(bob_records.len(), 1);
    assert_eq!(microcredits(&bob_records[0]), Some(amount));
    // The change and the new config return to `Y`, and decrypt with `y`.
    let vault_records = owned_records(&transaction, &vault_view_key);
    assert_eq!(vault_records.len(), 2);
    let change = vault_records.iter().find_map(microcredits).unwrap();
    assert_eq!(change, 3_000_000 - amount);
    // `y` also decrypts the vault's transitions: `tvk = (y * tpk).x`.
    for transition in transaction.transitions().filter(|t| t.function_name().to_string() != "fee_public") {
        let tvk = (*transition.tpk() * *vault_view_key).to_x_coordinate();
        assert_eq!(*transition.tcm(), <CurrentNetwork as Network>::hash_psd2(&[tvk]).unwrap());
    }

    // An ordinary execution still works after the vault's executions.
    let inputs = [Value::from(Literal::Address(bob_address)), Value::from(Literal::U64(U64::new(1)))];
    let transaction = vm
        .execute(&genesis, ("credits.aleo", "transfer_public_to_private"), inputs.iter(), None, 0, None, rng)
        .unwrap();
    assert_all_accepted(&add_block(&vm, &genesis, &[transaction], rng), 1);
}
