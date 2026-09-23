# Confirmation review: energy selection

Final status: **unconfirmed**. The original measured search selected `vector_add-n67108864-b256` and returned exit status 0 under its then-current Pareto-only confirmation protocol. This addendum adds a b512 capture and evaluates both speed-eligible candidates under the current discovery-or-confirmation MAD rule. The original `report.json` and raw captures remain unchanged.

| Candidate | Confirmed runtime (ms/launch) | Runtime MAD (ms) | Confirmed energy (J/launch) | Energy MAD (J) |
|---|---:|---:|---:|---:|
| vector_add-n67108864-b256 | 0.288351 | 0.000115 | 0.13698511 | 0.00005227 |
| vector_add-n67108864-b512 | 0.296487 | 0.000086 | 0.13821939 | 0.00021694 |

In the original discovery data, b256 and b512 differed by 0.00014809 J/launch against summed MAD 0.00064312 J, so they overlap. The follow-up confirmation gap is 0.00123428 J/launch against summed confirmation MAD 0.00026921 J. Although the independent confirmation medians separate, the predeclared follow-up rule treats candidates as indistinguishable if either discovery or confirmation overlaps; therefore this evidence makes no unique energy-minimum claim. Both candidates passed their 20% repeatability checks, and b256 remained within 5% of the confirmed fastest. It is the fastest practical recommendation in the tied set.

This MAD comparison is descriptive, not a statistical significance test. The b512 follow-up raw power traces, summaries, worker output and manifest are in `followup-b512/capture/`; the independent b256 confirmation is in `confirmations/vector_add-n67108864-b256/`.
