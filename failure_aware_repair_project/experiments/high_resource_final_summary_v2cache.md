# High_resource final simulator summary

- Topology: 4 spine, 8 leaf, 8 servers/leaf, 64 servers total.
- High-resource protection: 8 protection servers total, one per leaf.
- Each tenant count uses 10 simulator-validated failure cases. Story search time is not counted as algorithm runtime.
- `switch_servers` is the number of unique protection servers occupied by the strategy mapping.
- Physical audit checks every trial and every strategy mapping for duplicate active physical server occupancy and working-to-working moves.

| tenants | strategy | avg JCT mean | makespan mean | switch servers mean | switch servers max | gain vs failover mean | seeds |
|---:|---|---:|---:|---:|---:|---:|---|
| 3 | failover | 116.141756 | 153.938262 | 1.000 | 1 | 0.000% | 3097,3084,3015,3039,3062,3012,3021,3027,3062,3074 |
| 3 | local | 86.599918 | 118.817037 | 1.600 | 3 | 26.136% | 3097,3084,3015,3039,3062,3012,3021,3027,3062,3074 |
| 3 | coordinate | 63.197333 | 83.136006 | 3.300 | 5 | 44.281% | 3097,3084,3015,3039,3062,3012,3021,3027,3062,3074 |
| 3 | story_delta |  |  |  |  | local=26.136%, coordinate=44.281%, coord-extra=18.146% |  |
| 4 | failover | 154.751651 | 191.127921 | 1.000 | 1 | 0.000% | 4012,4012,4064,4064,4064,4064,4040,4012,4012,4018 |
| 4 | local | 123.755038 | 171.078425 | 1.200 | 2 | 19.709% | 4012,4012,4064,4064,4064,4064,4040,4012,4012,4018 |
| 4 | coordinate | 79.329657 | 96.762334 | 3.500 | 6 | 48.460% | 4012,4012,4064,4064,4064,4064,4040,4012,4012,4018 |
| 4 | story_delta |  |  |  |  | local=19.709%, coordinate=48.460%, coord-extra=28.751% |  |
| 5 | failover | 135.763614 | 189.780324 | 1.000 | 1 | 0.000% | 5019,5054,5014,5014,5054,5054,5042,5014,5113,5019 |
| 5 | local | 110.972302 | 146.621070 | 1.500 | 2 | 18.282% | 5019,5054,5014,5014,5054,5054,5042,5014,5113,5019 |
| 5 | coordinate | 87.477348 | 124.423732 | 4.000 | 5 | 35.653% | 5019,5054,5014,5014,5054,5054,5042,5014,5113,5019 |
| 5 | story_delta |  |  |  |  | local=18.282%, coordinate=35.653%, coord-extra=17.371% |  |
| 6 | failover | 135.296870 | 198.399092 | 1.000 | 1 | 0.000% | 6101,6012,6101,6101,6101,6089,6101,6041,6101,6017 |
| 6 | local | 112.183121 | 170.093118 | 1.500 | 2 | 16.748% | 6101,6012,6101,6101,6101,6089,6101,6041,6101,6017 |
| 6 | coordinate | 87.635649 | 109.868984 | 4.200 | 7 | 34.870% | 6101,6012,6101,6101,6101,6089,6101,6041,6101,6017 |
| 6 | story_delta |  |  |  |  | local=16.748%, coordinate=34.870%, coord-extra=18.122% |  |
| 7 | failover | 124.253110 | 195.893107 | 1.000 | 1 | 0.000% | 7010,7027,7119,7027,7027,7094,7010,7010,7010,7010 |
| 7 | local | 107.529921 | 180.065138 | 1.400 | 2 | 12.707% | 7010,7027,7119,7027,7027,7094,7010,7010,7010,7010 |
| 7 | coordinate | 83.564921 | 112.948608 | 4.500 | 6 | 31.956% | 7010,7027,7119,7027,7027,7094,7010,7010,7010,7010 |
| 7 | story_delta |  |  |  |  | local=12.707%, coordinate=31.956%, coord-extra=19.250% |  |
| 8 | failover | 136.969840 | 203.555424 | 1.000 | 1 | 0.000% | 8049,8022,8049,8096,8036,8118,8071,8049,8049,8049 |
| 8 | local | 112.671601 | 159.461931 | 1.500 | 2 | 17.378% | 8049,8022,8049,8096,8036,8118,8071,8049,8049,8049 |
| 8 | coordinate | 90.955375 | 129.412582 | 4.800 | 6 | 33.382% | 8049,8022,8049,8096,8036,8118,8071,8049,8049,8049 |
| 8 | story_delta |  |  |  |  | local=17.378%, coordinate=33.382%, coord-extra=16.004% |  |

## Protection Occupancy Audit

- Overall audit: PASS; checked 60 trials and 180 strategy mappings.
- tenants=3: PASS, max used protection servers failover=1, local=3, coordinate=5; violations=0.
- tenants=4: PASS, max used protection servers failover=1, local=2, coordinate=6; violations=0.
- tenants=5: PASS, max used protection servers failover=1, local=2, coordinate=5; violations=0.
- tenants=6: PASS, max used protection servers failover=1, local=2, coordinate=7; violations=0.
- tenants=7: PASS, max used protection servers failover=1, local=2, coordinate=6; violations=0.
- tenants=8: PASS, max used protection servers failover=1, local=2, coordinate=6; violations=0.
