# High_resource_homo final simulator summary

- Topology: 4 spine, 8 leaf, 8 servers/leaf, 64 servers total.
- High-resource protection: 8 protection servers total, one per leaf.
- Each tenant count uses 10 simulator-validated failure cases. Story search time is not counted as algorithm runtime.
- `switch_servers` is the number of unique protection servers occupied by the strategy mapping.
- Physical audit checks every trial and every strategy mapping for duplicate active physical server occupancy and working-to-working moves.
- Homogeneous workload: all tenants use GPT13B dominant trace-derived collective.

| tenants | strategy | avg JCT mean | makespan mean | switch servers mean | switch servers max | gain vs failover mean | seeds |
|---:|---|---:|---:|---:|---:|---:|---|
| 3 | failover | 29.888116 | 34.174718 | 1.000 | 1 | 0.000% | 3074,3027,3012,3074,3062,3016,3097,3039,3084,3015 |
| 3 | local | 21.315099 | 28.626740 | 2.000 | 3 | 28.465% | 3074,3027,3012,3074,3062,3016,3097,3039,3084,3015 |
| 3 | coordinate | 17.956207 | 20.219407 | 2.800 | 4 | 39.670% | 3074,3027,3012,3074,3062,3016,3097,3039,3084,3015 |
| 3 | story_delta |  |  |  |  | local=28.465%, coordinate=39.670%, coord-extra=11.205% |  |
| 4 | failover | 33.255018 | 39.887217 | 1.000 | 1 | 0.000% | 4053,4088,4114,4088,4028,4064,4012,4103,4076,4064 |
| 4 | local | 24.804046 | 33.725247 | 1.600 | 3 | 24.578% | 4053,4088,4114,4088,4028,4064,4012,4103,4076,4064 |
| 4 | coordinate | 19.026904 | 23.251212 | 3.200 | 5 | 41.915% | 4053,4088,4114,4088,4028,4064,4012,4103,4076,4064 |
| 4 | story_delta |  |  |  |  | local=24.578%, coordinate=41.915%, coord-extra=17.337% |  |
| 5 | failover | 32.832584 | 37.388573 | 1.000 | 1 | 0.000% | 5090,5054,5066,5054,5113,5014,5014,5014,5077,5101 |
| 5 | local | 29.346937 | 33.991700 | 1.000 | 1 | 10.087% | 5090,5054,5066,5054,5113,5014,5014,5014,5077,5101 |
| 5 | coordinate | 21.502619 | 28.186858 | 4.100 | 5 | 34.255% | 5090,5054,5066,5054,5113,5014,5014,5014,5077,5101 |
| 5 | story_delta |  |  |  |  | local=10.087%, coordinate=34.255%, coord-extra=24.168% |  |
| 6 | failover | 36.127743 | 46.515213 | 1.000 | 1 | 0.000% | 6012,6031,6031,6043,6015,6042,6064,6012,6017,6017 |
| 6 | local | 29.980113 | 34.501748 | 1.300 | 2 | 16.808% | 6012,6031,6031,6043,6015,6042,6064,6012,6017,6017 |
| 6 | coordinate | 25.789160 | 33.424576 | 4.600 | 7 | 28.417% | 6012,6031,6031,6043,6015,6042,6064,6012,6017,6017 |
| 6 | story_delta |  |  |  |  | local=16.808%, coordinate=28.417%, coord-extra=11.609% |  |
| 7 | failover | 35.153516 | 44.460507 | 1.000 | 1 | 0.000% | 7084,7084,7060,7010,7060,7060,7010,7027,7027,7027 |
| 7 | local | 30.324414 | 37.843874 | 1.400 | 3 | 13.585% | 7084,7084,7060,7010,7060,7060,7010,7027,7027,7027 |
| 7 | coordinate | 26.512750 | 34.121286 | 4.000 | 6 | 24.437% | 7084,7084,7060,7010,7060,7060,7010,7027,7027,7027 |
| 7 | story_delta |  |  |  |  | local=13.585%, coordinate=24.437%, coord-extra=10.853% |  |
| 8 | failover | 33.830276 | 43.574626 | 1.000 | 1 | 0.000% | 8049,8049,8049,8049,8049,8049,8036,8109,8049,8071 |
| 8 | local | 29.689043 | 34.806860 | 1.300 | 2 | 11.961% | 8049,8049,8049,8049,8049,8049,8036,8109,8049,8071 |
| 8 | coordinate | 25.459619 | 31.256886 | 4.800 | 6 | 24.400% | 8049,8049,8049,8049,8049,8049,8036,8109,8049,8071 |
| 8 | story_delta |  |  |  |  | local=11.961%, coordinate=24.400%, coord-extra=12.439% |  |

## Protection Occupancy Audit

- Overall audit: PASS; checked 60 trials and 180 strategy mappings.
- tenants=3: PASS, max used protection servers failover=1, local=3, coordinate=4; violations=0.
- tenants=4: PASS, max used protection servers failover=1, local=3, coordinate=5; violations=0.
- tenants=5: PASS, max used protection servers failover=1, local=1, coordinate=5; violations=0.
- tenants=6: PASS, max used protection servers failover=1, local=2, coordinate=7; violations=0.
- tenants=7: PASS, max used protection servers failover=1, local=3, coordinate=6; violations=0.
- tenants=8: PASS, max used protection servers failover=1, local=2, coordinate=6; violations=0.
