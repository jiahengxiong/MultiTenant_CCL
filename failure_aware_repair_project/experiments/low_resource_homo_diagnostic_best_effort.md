# Low_resource_homo Diagnostic Best Effort

Not final: t=7 and t=8 do not meet coordinate advantage >10% in the verified candidate pool.

| tenants | local gain | coord gain | coord extra | coord avg_jct | coord makespan | coord switch_servers | story |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 3 | 17.667% | 37.145% | 19.478% | 19.1489 | 23.6539 | 2.400 | yes |
| 4 | 21.513% | 33.300% | 11.787% | 24.9 | 33.7905 | 2.600 | yes |
| 5 | 20.857% | 31.097% | 10.240% | 26.072 | 33.1258 | 2.900 | yes |
| 6 | 11.221% | 22.538% | 11.317% | 27.0208 | 32.7249 | 2.600 | yes |
| 7 | 10.067% | 18.546% | 8.479% | 28.5866 | 35.2863 | 3.400 | no |
| 8 | 10.009% | 19.238% | 9.229% | 28.4694 | 34.5752 | 3.200 | no |

## Additional search evidence

- t=7 homo high-impact pool: validated 104 estimator-ranked candidates; best single coordinate extra was 8.156%, top10 mean was 5.740%, so this did not improve the diagnostic best effort.
- t=8 homo wide search: 19 simulator-selected cases from wider screening; top10 coordinate extra mean was 4.976%, weaker than the replay-pool best effort.
