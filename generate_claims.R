library(SynthETIC)

# Load the built-in test claims object (already has all parameters set up)
data(test_claims_object)

# Generate transaction data directly from it
transaction_data <- generate_transaction_dataset(test_claims_object)

# Check what you got
dim(transaction_data)
head(transaction_data, 30)

# Export
write.csv(transaction_data, "data/synthetic_transactions.csv", row.names = FALSE)

print(paste("✓ Generated", nrow(transaction_data), "transaction records"))




# library(SynthETIC)

# # Use the built-in test dataset
# data(test_claim_dataset)

# # This gives you realistic claim data
# head(test_claim_dataset)
# dim(test_claim_dataset)

# # Export it
# write.csv(test_claim_dataset, "data/synthetic_claims.csv", row.names = FALSE)

# print("✓ Done!")
