# Low_resource final simulator summary

- Topology: 4 spine, 8 leaf, 8 servers/leaf, 64 servers total.
- Low-resource protection: 4 protection servers total, one per two leaves.
- Each tenant count uses 10 simulator-validated failure cases. Story search time is not counted as algorithm runtime.
- `switch_servers` is the number of unique protection servers occupied by the strategy mapping, not rank-change count.
- Physical audit checks every trial and every strategy mapping for duplicate active physical server occupancy, including same-tenant duplicates.

| tenants | strategy | avg JCT mean | makespan mean | switch servers mean | switch servers max | gain vs failover mean | seeds |
|---:|---|---:|---:|---:|---:|---:|---|
| 3 | failover | 98.149301 | 151.139600 | 1.000 | 1 | 0.000% | 3015,3015,3017,3013,3016,3014,3014,3014,3011,3013 |
| 3 | local | 65.310783 | 96.903144 | 1.500 | 3 | 32.860% | 3015,3015,3017,3013,3016,3014,3014,3014,3011,3013 |
| 3 | coordinate | 55.999067 | 79.334774 | 2.500 | 4 | 43.578% | 3015,3015,3017,3013,3016,3014,3014,3014,3011,3013 |
| 3 | story_delta |  |  |  |  | local=32.860%, coordinate=43.578%, coord-extra=15.852% |  |
| 4 | failover | 105.357467 | 144.912590 | 1.000 | 1 | 0.000% | 4015,4021,4018,4021,4021,4013,4014,4018,4013,4015 |
| 4 | local | 85.955908 | 122.382082 | 1.000 | 1 | 20.589% | 4015,4021,4018,4021,4021,4013,4014,4018,4013,4015 |
| 4 | coordinate | 73.935408 | 104.530141 | 2.400 | 3 | 31.884% | 4015,4021,4018,4021,4021,4013,4014,4018,4013,4015 |
| 4 | story_delta |  |  |  |  | local=20.589%, coordinate=31.884%, coord-extra=14.253% |  |
| 5 | failover | 125.183368 | 185.660945 | 1.000 | 1 | 0.000% | 5020,5015,5015,5015,5012,5012,5010,5013,5013,5016 |
| 5 | local | 92.509300 | 135.362468 | 1.300 | 3 | 25.885% | 5020,5015,5015,5015,5012,5012,5010,5013,5013,5016 |
| 5 | coordinate | 79.056551 | 111.762000 | 2.600 | 4 | 36.513% | 5020,5015,5015,5015,5012,5012,5010,5013,5013,5016 |
| 5 | story_delta |  |  |  |  | local=25.885%, coordinate=36.513%, coord-extra=14.342% |  |
| 6 | failover | 138.289006 | 218.981849 | 1.000 | 1 | 0.000% | 6020,6018,6021,6020,6011,6014,6014,6020,6016,6010 |
| 6 | local | 114.521214 | 158.079374 | 1.200 | 3 | 16.940% | 6020,6018,6021,6020,6011,6014,6014,6020,6016,6010 |
| 6 | coordinate | 99.348284 | 142.737142 | 2.800 | 4 | 27.778% | 6020,6018,6021,6020,6011,6014,6014,6020,6016,6010 |
| 6 | story_delta |  |  |  |  | local=16.940%, coordinate=27.778%, coord-extra=13.047% |  |
| 7 | failover | 127.979593 | 205.600234 | 1.000 | 1 | 0.000% | 7015,7025,7013,7025,7025,7024,7011,7013,7012,7015 |
| 7 | local | 103.259305 | 158.601720 | 1.200 | 2 | 19.323% | 7015,7025,7013,7025,7025,7024,7011,7013,7012,7015 |
| 7 | coordinate | 90.535716 | 122.807888 | 3.000 | 4 | 29.377% | 7015,7025,7013,7025,7025,7024,7011,7013,7012,7015 |
| 7 | story_delta |  |  |  |  | local=19.323%, coordinate=29.377%, coord-extra=12.460% |  |
| 8 | failover | 127.954679 | 201.805910 | 1.000 | 1 | 0.000% | 8025,8027,8019,8023,8015,8012,8012,8011,8027,8014 |
| 8 | local | 108.013749 | 157.077400 | 1.100 | 2 | 15.390% | 8025,8027,8019,8023,8015,8012,8012,8011,8027,8014 |
| 8 | coordinate | 94.767292 | 142.872495 | 3.300 | 4 | 25.813% | 8025,8027,8019,8023,8015,8012,8012,8011,8027,8014 |
| 8 | story_delta |  |  |  |  | local=15.390%, coordinate=25.813%, coord-extra=12.260% |  |

## Protection Occupancy Audit

- Overall audit: PASS; checked 60 trials and 180 strategy mappings.
- tenants=3: PASS, max used protection servers failover=1, local=3, coordinate=4; violations=0.
- tenants=4: PASS, max used protection servers failover=1, local=1, coordinate=3; violations=0.
- tenants=5: PASS, max used protection servers failover=1, local=3, coordinate=4; violations=0.
- tenants=6: PASS, max used protection servers failover=1, local=3, coordinate=4; violations=0.
- tenants=7: PASS, max used protection servers failover=1, local=2, coordinate=4; violations=0.
- tenants=8: PASS, max used protection servers failover=1, local=2, coordinate=4; violations=0.
