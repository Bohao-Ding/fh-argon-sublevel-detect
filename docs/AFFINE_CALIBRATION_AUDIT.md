# Common affine calibration: second analysis

The current archive supports conditional effective-energy concentration. It does not independently measure an absolute voltage gain or establish a common instrument error. A common current readout correction was tested on frozen H1/G1/C responses. The prescribed inner-fold rule retained the identity readout in all four outer training sets and the final training set; the published physical kernel and spectra therefore remain unchanged.

## What was tested

The primary proposed correction is `I_measured = g * I_core + c`, followed by the existing nonnegative readout. `g,c` are estimated by range-normalized, equal-curve least squares **using training observations only**. The comparison is against `g=1,c=0`. This is a residual-response diagnostic, not a meter calibration against a known current source. The physical response already contains amplitude, background, periodic phase and a retarding scale; its fitted per-curve current gains and zeros are also retained in the old evidence.

For C, each outer fold retains its earlier training-only smoothing choice. Its three corresponding inner whole-curve folds compare identity and common affine readout using the original one-SE rule, preferring identity when prediction is comparable. Final selection uses four inner folds. This is a conditional reanalysis of frozen kernels, not a fresh joint search over kernel/readout hyperparameters. The SE rule is a simplicity heuristic across conditions, not a population significance test. No kernel checkpoint is retrained and no heldout observation enters its readout coefficients.

An additional four-coefficient representation, `(g0+g1*z)*I_core+c0+c1*z`, with `z=(Vr-4)/4`, is an **exploratory diagnostic only**. It is evaluated to distinguish a common readout error from missing condition dependence. It is never selected or substituted for the reported primary prediction. Final readout decisions freeze before unchanged 10 V stress evaluation.

## Results

Mean across four heldout conditions, after the median over the three existing optimization starts:

| Kernel | Identity NRMSE | Common affine NRMSE | Exploratory Vr-dependent NRMSE |
|---|---:|---:|---:|
| H1 | 0.108329 | 0.107887 | 0.066107 |
| G1 | 0.108245 | 0.107731 | 0.066180 |
| C | 0.122422 | 0.115699 | 0.080483 |

The common C readout improves the unselected diagnostic mean by 5.49%, but worsens the 4 V outer fold. All five inner decisions retain identity, so the nested selected C score remains 0.122422. Final common-readout estimates are model-dependent: H1 gives `g=1.00490,c=-0.03378 microampere`, while C gives `g=1.04315,c=-0.05948 microampere`. These are not verified electronics specifications.

The stronger Vr-dependent diagnostic gain indicates a systematic relationship between residuals and collection conditions. It does not identify an equipment fault: an underrepresented collection/background response produces the same effect. In the C outer 0 V fold, its inferred additive term is +0.25973 microampere despite the archived near-zero early current. This behavior and its exploratory status preclude adopting it as physical calibration.

The C 10 V stress NRMSE is 0.219618 without correction, 0.215728 with common readout, and 0.129468 with the exploratory Vr-dependent extrapolation. No 10 V observation fits those coefficients. This suggests that current/background mismatch contributes to the pressure-test error; it neither proves overheating nor repairs all shape information.

## Voltage origin and scale

Write `Va_actual = a*Va_read+b`. For the periodic component alone,

`D(a*Va+b; E, phi, width) = D(Va; E/a, (b+phi)/a, width/a)`.

Thus a constant voltage origin is absorbed in phase, whereas an unknown voltage gain is confounded with the inferred energy and kernel width. This identity concerns the periodic response; it is not claimed to be an exact symmetry of every bounded background/collection term. Those empirical terms are not an independent voltage standard. Subtracting the mode's 0.212 eV excess from an accelerating-voltage origin would not correct the period. Mapping C's 12.04 eV mode into the known 4s range would require the hypothetical gain 0.95917–0.98240, but the archive contains no independent calibration data to determine it. Using the group itself as an anchor would produce a conditional calibrated result, not an independent verification of that group.

Matched troughs in 4/6/8 V have a shared visible period about 11.48–11.53 V, with a condition-dependent shift 0.438–0.469 V/V across Savitzky–Golay windows of 7/11/15 points. At the central 11-point setting, a common-position fit has RMS 0.79364 V; adding a linear Vr shift lowers it to 0.20958 V. Adding a condition-dependent spacing lowers it only to 0.20833 V and is sensitive to smoothing. These are descriptive feature fits on the 0.5 V sampled grid, with no absolute collision-order assignment or replicate-based confidence interval. The 0 V curve lacks four comparably prominent troughs and is not forced into this regression. The 10 V features are listed separately and do not set its coefficients.

## Consequence for the energy result

C remains an effective broad concentration: mode 12.04 eV, connected half-height region 11.48–12.62 eV, 48.7% effective mass in that lobe, and 11.2% in the narrow first 4s group. H1 remains 11.7636 eV. The broad overlap is meaningful at the intended teaching-experiment scale; this calibration audit supplies no independently measured scale with which to move the distribution into closer agreement. Condition-dependent phase/collection response is a more specific future modeling question than another redundant global voltage affine pair. No four-point spectrum is imposed.

## Reproduction and evidence

```powershell
python -B scripts/audit_affine_calibration.py --output output/calibration_audit_replay
python -B -m pytest -p no:cacheprovider tests/test_affine_readout.py
```

Use an empty output directory and remove disposable replay/test output after inspection. Formal local derived outputs remain at `output/calibration_audit_v1`. The independently replayable release is `source_data_package/calibration_audit_v1`; twelve additional parameter-only inner C references are at `source_data_package/calibration_reference_v1`. Their original unit identities were reconstructed from training bytes, initialization, settings and the five frozen kernel source hashes. Both new packages have actual-byte SHA-256 lists and provenance; the older spectral evidence is unchanged. All starts are computational initializations, not experimental repeats.

## Physical sources

- [University of Washington PHYS 432 laboratory guide, Sec. 2.2.4](https://courses.washington.edu/phys432/franck-hertz/franck-hertz.pdf): separate voltmeter observations establish the voltage gain and offset. Its particular apparatus scale is not assigned to this project.
- [MIT Junior Lab guide, Sec. VI](https://ocw.mit.edu/courses/8-13-14-experimental-physics-i-ii-junior-lab-fall-2016-spring-2017/afdfff9f8bbe067239af19c8b178a764_MIT8_13-14F16-S17exp7.pdf): contact potential affects voltage origin.
- [Rapior, Sengstock and Baev, AJP 74, 423–428 (2006)](https://doi.org/10.1119/1.2174033), [read full paper](https://www.physik.uni-jena.de/pafmedia/9356/paper-rapior-sengstock-baev.pdf?nonactive=1&suffix=pdf): Hg/Ne peak spacings and retarding shifts can reflect collision/collection physics. Applying this possibility to the present argon residuals is an inference, not a demonstrated identical mechanism.
- [NIST Ar I levels](https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable5.htm): external energy-range comparison only. Sources checked 2026-10-02.
