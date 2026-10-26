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

use super::*;

use snarkvm_synthesizer_snark::VerifyingKey;

/// Returns the public balance of the given address.
fn public_balance(vm: &VM<CurrentNetwork, LedgerType>, address: &Address<CurrentNetwork>) -> u64 {
    let value = vm
        .finalize_store()
        .get_value_confirmed(
            ProgramID::from_str("credits.aleo").unwrap(),
            Identifier::from_str("account").unwrap(),
            &Plaintext::from(Literal::Address(*address)),
        )
        .unwrap();
    match value {
        Some(Value::Plaintext(Plaintext::Literal(Literal::U64(balance), _))) => *balance,
        _ => 0,
    }
}

/// Returns a deployment transaction whose verifying keys underreport their number of constraints, so that
/// it fails verification. Its owner signature and fee are valid for its deployment ID.
fn sample_unverifiable_deployment(
    vm: &VM<CurrentNetwork, LedgerType>,
    private_key: &PrivateKey<CurrentNetwork>,
    rng: &mut TestRng,
) -> Transaction<CurrentNetwork> {
    let program = Program::from_str(
        r"
program constraint_underreport_v23.aleo;

function do:
    input r0 as u32.private;
    add r0 r0 into r1;
    output r1 as u32.public;",
    )
    .unwrap();

    // Create the deployment transaction.
    let transaction = vm.deploy(private_key, &program, None, 0, None, rng).unwrap();
    let Transaction::Deploy(_, _, _, deployment, fee) = transaction else {
        panic!("Expected a deployment transaction");
    };

    // Decrease the number of constraints in the verifying keys.
    let mut vks_with_underreport = Vec::with_capacity(deployment.verifying_keys().len());
    for (id, (vk, cert)) in deployment.verifying_keys() {
        let mut vk_deref = vk.deref().clone();
        vk_deref.circuit_info.num_constraints -= 2;
        let vk = VerifyingKey::new(Arc::new(vk_deref), vk.num_variables());
        vks_with_underreport.push((*id, (vk, cert.clone())));
    }

    // Create a new deployment transaction with the underreported verifying keys.
    let adjusted_deployment = Deployment::new(
        deployment.edition(),
        deployment.program().clone(),
        vks_with_underreport,
        deployment.program_checksum(),
        deployment.program_owner(),
    )
    .unwrap();
    let deployment_id = adjusted_deployment.to_deployment_id().unwrap();
    // Sign the adjusted deployment, and authorize a fee for it.
    let program_owner = ProgramOwner::new(private_key, deployment_id, rng).unwrap();
    let authorization =
        vm.authorize_fee_public(private_key, *fee.base_amount().unwrap(), 0, deployment_id, rng).unwrap();
    let fee = vm.execute_fee_authorization(authorization, None, rng).unwrap();
    let adjusted_transaction = Transaction::from_deployment(program_owner, adjusted_deployment, fee).unwrap();
    assert!(vm.check_transaction(&adjusted_transaction, None, rng).is_err());
    adjusted_transaction
}

#[test]
fn test_unverifiable_deployment_is_rejected_from_v23() {
    let rng = &mut TestRng::default();

    // Initialize the VM at `ConsensusVersion::V23`.
    let vm = sample_vm_at_height(CurrentNetwork::CONSENSUS_HEIGHT(ConsensusVersion::V23).unwrap(), rng);
    let private_key = sample_genesis_private_key(rng);
    let address = Address::try_from(&private_key).unwrap();

    // Sample a deployment that fails verification but whose fee is valid.
    let transaction = sample_unverifiable_deployment(&vm, &private_key, rng);

    // Create a block: the deployment is rejected with its fee charged, not aborted.
    let balance = public_balance(&vm, &address);
    let block = sample_next_block(&vm, &private_key, std::slice::from_ref(&transaction), rng).unwrap();
    assert_eq!(block.transactions().num_accepted(), 0);
    assert_eq!(block.transactions().num_rejected(), 1);
    assert!(block.aborted_transaction_ids().is_empty());
    let rejected = block.transactions().iter().next().unwrap();
    assert!(matches!(rejected, ConfirmedTransaction::RejectedDeploy(..)));
    assert_eq!(rejected.to_unconfirmed_transaction().unwrap(), transaction);

    // Add the block to the VM, and check the rejection reason, the fee, and that the program was not deployed.
    vm.add_next_block(&block).unwrap();
    let rejected_id = *rejected.id();
    assert_eq!(vm.finalize_store().get_rejected_reason(&rejected_id).unwrap(), Some(RejectedReason::Verification));
    assert_eq!(public_balance(&vm, &address), balance - *transaction.fee_amount().unwrap());
    assert!(!vm.contains_program(&ProgramID::from_str("constraint_underreport_v23.aleo").unwrap()));
}

#[test]
fn test_unverifiable_deployment_is_aborted_before_v23() {
    let rng = &mut TestRng::default();

    // Initialize the VM one block before `ConsensusVersion::V23`, so that the next block is before V23.
    let vm = sample_vm_at_height(CurrentNetwork::CONSENSUS_HEIGHT(ConsensusVersion::V23).unwrap() - 2, rng);
    let private_key = sample_genesis_private_key(rng);
    let address = Address::try_from(&private_key).unwrap();

    // Sample a deployment that fails verification but whose fee is valid.
    let transaction = sample_unverifiable_deployment(&vm, &private_key, rng);

    // Create a block: the deployment is aborted, and nothing is charged.
    let balance = public_balance(&vm, &address);
    let block = sample_next_block(&vm, &private_key, std::slice::from_ref(&transaction), rng).unwrap();
    assert!(CurrentNetwork::CONSENSUS_VERSION(block.height()).unwrap() < ConsensusVersion::V23);
    assert_eq!(block.transactions().len(), 0);
    assert_eq!(block.aborted_transaction_ids(), &[transaction.id()]);
    vm.add_next_block(&block).unwrap();
    assert_eq!(public_balance(&vm, &address), balance);
    assert!(vm.block_store().contains_transaction_id(&transaction.id()).unwrap());
}
