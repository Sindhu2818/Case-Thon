# Data Dictionary - with prediction-use and leakage verdicts

Derived from the provided `data_dictionary.csv`; verdicts apply our prediction-time rule: a feature may enter the model only if it was recorded at/before the teleconsultation instant of the episode being scored (or is a time-invariant reference attribute).

| table | column | meaning | use for prediction | leakage risk |
|---|---|---|---|---|
| patient_360_reference.csv | `patient_id` | Canonical synthetic patient identifier in the reference entity table. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `canonical_name` | Canonical synthetic patient name. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `gender` | Synthetic recorded sex/gender marker used for linkage context. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `date_of_birth` | Canonical synthetic date of birth. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `age_as_of_2026` | Age derived for 2026. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `masked_mobile` | Canonical phone with only last 4 digits visible. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `village_id` | Canonical synthetic village identifier. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `block` | Administrative block. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `district` | Synthetic-use district context. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `preferred_language` | Preferred interaction language. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `known_ncd_status` | Reference NCD status. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `vulnerability_group` | Synthetic operational vulnerability segment. | Yes - static reference | Time-invariant reference data. |
| patient_360_reference.csv | `household_id` | Synthetic household grouping identifier. | Yes - static reference | Time-invariant reference data. |
| teleconsultations.csv | `teleconsult_id` | Unique teleconsultation identifier. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `tele_source_patient_id` | Patient identifier used only by teleconsult system. *(pack note: No direct crosswalk is supplied to candidates.)* | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `gender` | Synthetic recorded sex/gender marker used for linkage context. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `age` | Source-record age; may differ slightly from reference. *(pack note: May contain deliberate source-system noise.)* | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `block` | Administrative block. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `district` | Synthetic-use district context. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `consult_date` | Date of teleconsultation. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `facility_id` | Synthetic facility identifier. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `chief_complaint` | Broad presenting complaint. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `diagnosis_group` | Broad diagnosis category. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `consult_mode` | Audio/video/assisted consultation mode. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `duration_min` | Consultation duration in minutes. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `medicine_advised` | Whether medicine was recommended. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `test_advised` | Whether diagnostic testing was recommended. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `review_advised` | Whether a review visit was recommended. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `review_due_days` | Recommended review interval in days. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `distance_to_facility_km` | Approximate synthetic road-independent distance proxy. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `preferred_language` | Preferred interaction language. | Yes - available at prediction time | Recorded at/before the consultation instant. |
| teleconsultations.csv | `connectivity_quality` | Recorded connectivity quality during consultation. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| teleconsultations.csv | `consult_status` | Consultation completion status. | Yes - available at prediction time | Recorded during the consultation (part of the encounter). |
| ncd_screening.csv | `ncd_record_id` | Unique NCD screening record. | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `ncd_source_patient_id` | Patient ID used by NCD system. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `gender` | Synthetic recorded sex/gender marker used for linkage context. | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `age` | Source-record age; may differ slightly from reference. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `block` | Administrative block. | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `district` | Synthetic-use district context. | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `screening_date` | NCD screening date. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `screening_type` | Where/how NCD screening occurred. | Conditional | Use only if strictly before the prediction timestamp. |
| ncd_screening.csv | `systolic_bp` | Synthetic systolic BP value. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `diastolic_bp` | Synthetic diastolic BP value. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `random_glucose_mg_dl` | Synthetic random blood glucose. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `bmi` | Synthetic BMI. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `ncd_status` | NCD condition recorded in the screening system. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `control_status` | Synthetic control category. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| ncd_screening.csv | `next_followup_due` | Recommended next NCD follow-up date. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `prescription_id` | Prescription identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `prescription_line_id` | Prescription medicine-line identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `rx_source_patient_id` | Patient ID used by prescription system. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `prescription_date` | Prescription date. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| prescriptions.csv | `teleconsult_id` | Unique teleconsultation identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `medicine_name` | Generic synthetic medicine name. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `medicine_class` | Medicine therapeutic class. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `frequency` | Simplified dose frequency. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `days_prescribed` | Intended number of treatment days. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `route` | Administration route. | Conditional | Use only if strictly before the prediction timestamp. |
| prescriptions.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `dispense_id` | Medicine dispensing event identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `pharm_source_patient_id` | Patient ID used by pharmacy register. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `dispense_date` | Date medicine was collected/attempted. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_dispensing.csv | `prescription_id` | Prescription identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `medicine_name` | Generic synthetic medicine name. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `prescribed_qty_proxy` | Simplified intended quantity proxy. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `dispensed_qty` | Quantity actually dispensed. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| medicine_dispensing.csv | `partial_fill` | 1 when only part of intended supply was dispensed. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| medicine_dispensing.csv | `stockout_flag` | 1 when stock-out affected the dispensing attempt. | Only as STRICTLY PRE-consult history | post-consult event; use monthly ledger instead |
| medicine_dispensing.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_dispensing.csv | `dispense_status` | Dispensed/partial/not dispensed due to stock. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| medicine_stock_status.csv | `stock_record_id` | Facility medicine stock snapshot identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_stock_status.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_stock_status.csv | `snapshot_month` | Month of stock snapshot. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_stock_status.csv | `medicine_name` | Generic synthetic medicine name. | Conditional | Use only if strictly before the prediction timestamp. |
| medicine_stock_status.csv | `opening_stock` | Opening stock units. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_stock_status.csv | `received_qty` | Units received during the month. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_stock_status.csv | `dispensed_qty_month` | Units dispensed in the stock ledger month. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_stock_status.csv | `closing_stock` | Closing stock units. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_stock_status.csv | `stockout_flag_month` | 1 if a stock-out occurred in the month. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_stock_status.csv | `stockout_days` | Number of stock-out days during the month. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| medicine_stock_status.csv | `stock_status` | Adequate/Low stock/Reorder raised. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| lab_tests.csv | `lab_record_id` | Lab/test record identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `lab_source_patient_id` | Patient ID used by lab system. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `order_date` | Test order date. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| lab_tests.csv | `test_name` | Ordered diagnostic test. | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `sample_date` | Sample/test date if completed. | Only as STRICTLY PRE-consult history | post-consult event (unless historical) |
| lab_tests.csv | `result_date` | Result date if available. | Only as STRICTLY PRE-consult history | post-consult event (unless historical) |
| lab_tests.csv | `result_flag` | Simplified normal/abnormal/critical result category. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| lab_tests.csv | `test_status` | Available or Ordered-not-completed. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| lab_tests.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| lab_tests.csv | `lab_type` | Facility/hub/external lab source. | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `visit_id` | Visit/encounter identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `visit_source_patient_id` | Patient ID used by visit register. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `visit_date` | Encounter date. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| followup_visits.csv | `visit_type` | Encounter type. | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `episode_id` | Synthetic post-teleconsult care-journey identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| followup_visits.csv | `clinical_status` | Simplified encounter outcome. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| followup_visits.csv | `referral_flag` | Whether referral/escalation occurred. | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `visit_id` | Visit/encounter identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `visit_source_patient_id` | Patient ID used by visit register. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `visit_date` | Encounter date. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| visit_history.csv | `visit_type` | Encounter type. | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `episode_id` | Synthetic post-teleconsult care-journey identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| visit_history.csv | `clinical_status` | Simplified encounter outcome. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| visit_history.csv | `referral_flag` | Whether referral/escalation occurred. | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `outreach_id` | ASHA/CHO outreach event identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `outreach_source_patient_id` | Patient ID used by outreach register. *(pack note: No direct crosswalk is supplied to candidates.)* | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `patient_name` | Source-record patient name; deliberately noisy. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `mobile` | Source-record phone; may be missing, mistyped or formatted differently. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `action_date` | Outreach date. | Only as strictly pre-consult history | Time-gate: date <= consult_date of the predicted episode. |
| outreach_actions.csv | `action_method` | Phone/home visit/message/village health day reminder. | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `contact_outcome` | Outcome of outreach attempt. | Only as STRICTLY PRE-consult history | post-consult outcome (unless historical) |
| outreach_actions.csv | `cadre` | ASHA or CHO. | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `facility_id` | Synthetic facility identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `episode_id` | Synthetic post-teleconsult care-journey identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| outreach_actions.csv | `followup_reason` | Reason for outreach. | Conditional | Use only if strictly before the prediction timestamp. |
| facility_reference.csv | `facility_id` | Synthetic facility identifier. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `facility_name` | Synthetic facility name. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `facility_type` | PHC or HWC. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `district` | Synthetic-use district context. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `block` | Administrative block. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `latitude` | Synthetic latitude for competition visualization only. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `longitude` | Synthetic longitude for competition visualization only. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `network_context` | Facility connectivity context. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `cho_count` | Synthetic number of CHOs. | Yes - static reference | Time-invariant reference data. |
| facility_reference.csv | `asha_linked_count` | Synthetic number of linked ASHAs. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `village_id` | Canonical synthetic village identifier. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `village` | Village name; source tables may contain formatting/spelling noise. *(pack note: May contain deliberate source-system noise.)* | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `district` | Synthetic-use district context. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `block` | Administrative block. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `latitude` | Synthetic latitude for competition visualization only. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `longitude` | Synthetic longitude for competition visualization only. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `population` | Synthetic village population. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `road_access` | Synthetic access condition. | Yes - static reference | Time-invariant reference data. |
| geography_reference.csv | `mobile_connectivity` | Synthetic village connectivity category. | Yes - static reference | Time-invariant reference data. |
| episode_outcomes.csv | `episode_id` | Synthetic post-teleconsult care-journey identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| episode_outcomes.csv | `teleconsult_id` | Unique teleconsultation identifier. | Conditional | Use only if strictly before the prediction timestamp. |
| episode_outcomes.csv | `consult_date` | Date of teleconsultation. | Conditional | Use only if strictly before the prediction timestamp. |
| episode_outcomes.csv | `cohort` | DEVELOPMENT or EVALUATION. | Split definition only | Defines DEVELOPMENT vs EVALUATION. |
| episode_outcomes.csv | `lost_to_followup_label` | Development label: 1 if any required care component remained incomplete; 0 otherwise. *(pack note: Blank for EVALUATION cohort.)* | LABEL - training only, never a feature | Outcome of the whole episode; defined only after the journey ends. |
| episode_outcomes.csv | `dropout_stage_label` | Development label for first unresolved stage; blank for evaluation cohort. *(pack note: Blank for EVALUATION cohort.)* | LABEL (stage) - training only, never a feature | First unresolved stage; post-consult information. |

## Labels

- `lost_to_followup_label` = 1 iff any advised component remained incomplete (matches stage != 'Completed care journey'; dev agreement 100%).
- `dropout_stage_label` = first unresolved stage in the order medicine -> test -> review.
