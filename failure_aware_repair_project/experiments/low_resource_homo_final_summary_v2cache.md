# Low_resource_homo Final Summary

Selection: strict simulator-validated 10-trial averages; local vs failover >10% and coordinate extra advantage over local >10%. Story search is scenario selection and is not counted as algorithm runtime.

| tenants | local gain | coord gain | coord extra | failover switch | local switch | coordinate switch | coord avg_jct | coord makespan | story |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 3 | 17.667% | 37.145% | 19.478% | 1.000 | 1.600 | 2.400 | 19.1489 | 23.6539 | yes |
| 4 | 21.513% | 33.300% | 11.787% | 1.000 | 1.000 | 2.600 | 24.9000 | 33.7905 | yes |
| 5 | 20.857% | 31.097% | 10.240% | 1.000 | 1.000 | 2.900 | 26.0720 | 33.1258 | yes |
| 6 | 11.221% | 22.538% | 11.317% | 1.000 | 1.200 | 2.600 | 27.0208 | 32.7249 | yes |
| 7 | 10.426% | 20.688% | 10.261% | 1.000 | 1.900 | 3.500 | 28.1762 | 35.4311 | yes |
| 8 | 10.077% | 20.228% | 10.151% | 1.000 | 2.200 | 3.200 | 27.7546 | 34.5516 | yes |

Audit: ok=True, checked_trials=60, checked_strategy_mappings=180, protection_pool_size=4.
