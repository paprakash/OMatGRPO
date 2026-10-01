## Workflow

1. Generate trajectory with current model → logp_old (integrate_with_logprob)
2. Store: (states, actions, rewards, logp_old)
3. Update model weights via gradient descent (multiple epochs)
4. For each update:
   - Recompute logp_new on stored trajectory (step_logprob)
   - Calculate ratio = exp(logp_new - logp_old)
   - Compute PPO loss with clipping
   - Backprop and update
5. Repeat from step 4 (K epochs on same data)
6. Discard old trajectory, goto step 1
