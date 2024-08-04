# Todo

## Sparsifying Convolutions (SpConv)
### Zero Blockers
2. Implement output to CSRPyr (forward only)
3. Implement 1x1 convolution system internal to the kernel (forward only)
4. Implement backward operation for pyramidal convolution
5. Implement contribution calculation for pyramidal convolution

### One Blocker
6. Add CSRPyr to display lib and use to ensure output is correct [Blockers: task 2]
7. Implement deconv to dense operation (forward only) [Blockers: task 2]
8. Convert 1x1 convolution system into LSTM and feed CSRPyr back into LSTM as state [Blocker: task 3]
9. Implement backward operation for output to CSRPyr [Blockers: task 2]
10. Implement contribution calculation for output to CSRPyr [Blockers: task 2]
11. Implement backward operation for 1x1 convolution system [Blockers: task 3]
12. Implement contribution calculation for 1x1 convolution system [Blockers: task 3]

### Two Blockers
13. Implement backward operation for deconv to dense [Blockers: task 7]
14. Implement contribution calculation for deconv to dense [Blockers: task 7]
15. Implement backward operation for LSTM with CSRPyr state [Blockers: task 8]
16. Implement contribution calculation for LSTM with CSRPyr state [Blockers: task 8]

### Three Blockers
17. Add variational aspect to middle layer as part of backward [Blockers: task 13]
18. Add attractor aspect to middle layer as part of backward [Blockers: task 13]

## Sparse Transformer Layers (SpXfmr) [Blockers: SpConv, needed for input]