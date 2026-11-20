# v0 deviations from the pre-registration

1. The first launch of the nine confirmatory runs was stopped after about two minutes because its progress log read
   the first 500 test puzzles. It was patched to log on the validation split and relaunched from scratch. No decision
   used those two minutes. `MANIFEST.sha256` holds the hashes of the relaunched code.
2. The `spike` seed-3 run was cut off by a job time limit near step 1,000 and rerun from scratch, identical settings.
3. H3 for the `spike` arm: seed 2 ran on the cloud CPU sandbox; seeds 1 and 3 ran on a laptop CPU (Windows,
   PyTorch 2.14.1) with the same frozen `bitloop.py` and `data.npz` (SHA-256 verified), because the sandbox session
   could not hold all three. Their logs are `runs/spike_k8_1.log` and `runs/spike_k8_3.log`.
4. Two exploratory runs at learning rate 8e-3 (`explore/`) were chosen after seeing the confirmatory result. They use
   the validation split and one seed, and support no confirmatory claim.
