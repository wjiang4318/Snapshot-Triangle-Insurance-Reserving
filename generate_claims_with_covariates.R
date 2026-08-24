#https://cran.r-project.org/web/packages/SynthETIC/vignettes/SynthETIC-covariates-demo.html
library(SynthETIC)
set.seed(20200131)

set_parameters(ref_claim = 200000, time_unit = 1/4)
time_unit <- return_parameters()[2]

years <- 10
I <- years / time_unit
E <- c(rep(12000, I))      # effective annual exposure rates
lambda <- c(rep(0.03, I))

# --- Steps 1-2: frequency, occurrence, base claim size ---
n_vector <- claim_frequency(I = I, E = E, freq = lambda)
occurrence_times <- claim_occurrence(frequency_vector = n_vector)
claim_sizes <- claim_size(frequency_vector = n_vector)

# --- Apply built-in covariates: Legal Representation, Injury Severity, Age of Claimant ---
covariates_obj <- test_covariates_obj

# The package's default severity->cost relativity table has severity 6 at
# 0.4 -- LOWER than severity 1's 0.6, and a sharp break from the otherwise clean
# monotonic climb across levels 1-5 (0.6, 1.2, 2.5, 5.0, 8.0). Overridden here to
# 13, continuing that same roughly-doubling trend, so severity 6 is genuinely
# the most severe/costly level as the name implies.
sev_relativity <- covariates_obj$relativity_sev
sev6_row <- which(sev_relativity$factor_i == "Injury Severity" &
                  sev_relativity$factor_j == "Injury Severity" &
                  sev_relativity$level_ik == "6")
covariates_obj$relativity_sev[sev6_row, "relativity"] <- 13

claim_size_covariates <- claim_size_adj(covariates_obj, claim_sizes, random_seed = 20200131)
claim_size_w_cov <- claim_size_covariates$claim_size_adj
covariates_data_obj <- claim_size_covariates$covariates_data

# --- Steps 3-8, using covariate-adjusted claim sizes throughout, package defaults ---
notidel <- claim_notification(n_vector, claim_size_w_cov)
setldel <- claim_closure(n_vector, claim_size_w_cov)
no_payments <- claim_payment_no(n_vector, claim_size_w_cov)
payment_sizes <- claim_payment_size(n_vector, claim_size_w_cov, no_payments)
payment_delays <- claim_payment_delay(n_vector, claim_size_w_cov, no_payments, setldel)
payment_times <- claim_payment_time(n_vector, occurrence_times, notidel, payment_delays)

demo_rate <- (1 + 0.02)^(1/4) - 1
base_inflation_vector <- rep(demo_rate, times = 2 * I)
payment_inflated <- claim_payment_inflation(
  n_vector, payment_sizes, payment_times, occurrence_times,
  claim_size_w_cov, base_inflation_vector
)

# --- Bundle into a claims object, generate the transaction-level dataset ---
all_claims <- claims(
  frequency_vector = n_vector,
  occurrence_list = occurrence_times,
  claim_size_list = claim_size_w_cov,
  notification_list = notidel,
  settlement_list = setldel,
  no_payments_list = no_payments,
  payment_size_list = payment_sizes,
  payment_delay_list = payment_delays,
  payment_time_list = payment_times,
  payment_inflated_list = payment_inflated
)
transaction_dataset <- generate_transaction_dataset(all_claims, adjust = FALSE)

# --- Attach covariates (claim_no order == covariates_data_obj$data row order) ---
cov_df <- data.frame(covariates_data_obj$data)
colnames(cov_df) <- c("legal_representation", "injury_severity", "age_of_claimant")
cov_df$claim_no <- seq_len(nrow(cov_df))

transaction_dataset_full <- merge(transaction_dataset, cov_df, by = "claim_no")
transaction_dataset_full <- transaction_dataset_full[order(
  transaction_dataset_full$claim_no, transaction_dataset_full$pmt_no), ]

# claims-level CSV is NOT written -- every claim-level field (incl. no_payment via
# count(pmt_no)) is fully derivable from the transaction-level file, verified by
# a 0-mismatch reconciliation check during development. Single source of truth.
dir.create("data", showWarnings = FALSE)
write.csv(transaction_dataset_full, "data/synthetic_transactions_with_covariates.csv",
          row.names = FALSE)

cat("Generated", nrow(transaction_dataset_full), "transaction records for",
    length(unique(transaction_dataset_full$claim_no)), "claims\n")
