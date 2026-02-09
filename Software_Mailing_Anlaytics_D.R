library(caret)
library(pROC)
library(dplyr)
library(ggplot2)
library(scales)
library(randomForest)
library(cluster)


northpoint <- read.csv("software_mailing_list.csv")

str(northpoint)
summary(northpoint)

#check missing values for all columns
colSums(is.na(northpoint))

#handle missing numeric values (median imputation)
numeric_cols <- c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")
for(col in numeric_cols){
  northpoint[[col]][is.na(northpoint[[col]])] <- median(northpoint[[col]], na.rm = TRUE)
}

#handle missing binary values (treat NA as 0)
binary_cols <- c("US", "Web.order", "Gender.male", "Address_is_res",
                 "source_a","source_b","source_c","source_d","source_e",
                 "source_m","source_o","source_h","source_r","source_s",
                 "source_t","source_u","source_p","source_x","source_w","Purchase")
for(col in binary_cols){
  northpoint[[col]][is.na(northpoint[[col]])] <- 0
}

#Check zero values in numeric columns
sapply(northpoint[, numeric_cols], function(x) sum(x == 0))


##EDA
#Filter only purchasers
purchasers <- subset(northpoint, Purchase == 1)

#Histogram of Freq (number of purchases)
hist(purchasers$Freq, breaks = seq(0, max(purchasers$Freq)+1, 1), 
     col = "lightcoral", main = "Purchasers: Number of Past Purchases", 
     xlab = "Number of Purchases", ylab = "Count")

#Add data labels above bars
freq_table <- table(purchasers$Freq)
text(x = as.numeric(names(freq_table)), y = freq_table + 2, labels = freq_table, pos = 3)

#Boxplot of Spending
boxplot(purchasers$Spending, main = "Purchasers: Spending Distribution", 
        ylab = "Spending ($)", col = "lightseagreen")
#Add median label
mtext(paste("Median:", median(purchasers$Spending)), side=3)

#Bar plots for acquisition sources (descending)
source_cols <- grep("^source_", names(purchasers), value = TRUE)
source_sums <- colSums(purchasers[, source_cols])
source_sums <- sort(source_sums, decreasing = TRUE) # descending order

barplot(source_sums, main = "Purchasers by Acquisition Source", 
        col = "lightsteelblue", las=2, cex.names = 0.8, ylab = "Number of Purchasers")
#Add data labels
text(x = seq_along(source_sums), y = source_sums + 2, labels = source_sums, pos = 3, cex=0.8)

#Purchasers by Gender (Other vs Male)
purchasers$Gender.label <- ifelse(purchasers$Gender.male == 1, "Male", "Other")
gender_count <- table(purchasers$Gender.label)
gender_avg <- tapply(purchasers$Spending, purchasers$Gender.label, mean)

bar_matrix <- rbind(gender_count, gender_avg)
barplot(bar_matrix, beside=TRUE, col=c("pink","green"),
        main="Purchasers: Count & Avg Spending by Gender", ylab="Value")
text(x = seq_len(ncol(bar_matrix))*1.5 - 0.5, y = bar_matrix, labels = round(bar_matrix,1), pos = 3)
legend("topright", legend=c("Count","Avg Spending"), fill=c("pink","green"))

#Purchasers by US (in vs out)
us_count <- table(purchasers$US)
us_avg <- tapply(purchasers$Spending, purchasers$US, mean)
bar_matrix <- rbind(us_count, us_avg)
barplot(bar_matrix, beside=TRUE, names.arg=c("Non-US","US"),
        col=c("gold","brown"), main="Purchasers: Count & Avg Spending by US", ylab="Value")
text(x = seq_len(ncol(bar_matrix))*1.5 - 0.5, y = bar_matrix, labels = round(bar_matrix,1), pos = 3)
legend("topright", legend=c("Count","Avg Spending"), fill=c("gold","brown"))

#Purchasers by Address type
addr_count <- table(purchasers$Address_is_res)
addr_avg <- tapply(purchasers$Spending, purchasers$Address_is_res, mean)
bar_matrix <- rbind(addr_count, addr_avg)
barplot(bar_matrix, beside=TRUE, names.arg=c("Non-Res","Res"),
        col=c("lightblue","maroon"), main="Purchasers: Count & Avg Spending by Address", ylab="Value")
text(x = seq_len(ncol(bar_matrix))*1.5 - 0.5, y = bar_matrix, labels = round(bar_matrix,1), pos = 3)
legend("topright", legend=c("Count","Avg Spending"), fill=c("skyblue","maroon"))

#Purchasers by Web Order
web_count <- table(purchasers$Web.order)
web_avg <- tapply(purchasers$Spending, purchasers$Web.order, mean)
bar_matrix <- rbind(web_count, web_avg)
barplot(bar_matrix, beside=TRUE, names.arg=c("No","Yes"),
        col=c("skyblue","lightgreen"), main="Purchasers: Count & Avg Spending by Web Order", ylab="Value")
text(x = seq_len(ncol(bar_matrix))*1.5 - 0.5, y = bar_matrix, labels = round(bar_matrix,1), pos = 3)
legend("topright", legend=c("Count","Avg Spending"), fill=c("skyblue","lightgreen"))



#Filter only purchasers
purchasers <- subset(northpoint, Purchase == 1)

#Count for each source
source_counts <- colSums(purchasers[, c("source_a","source_u","source_w","source_e")])
source_counts

# Optional: percentage out of total purchasers
source_percent <- round(100 * source_counts / nrow(purchasers), 1)
source_percent


#draft2
#step4: Predictor Analysis and Relevancy
#Predictor Analysis for Clustering (Focus on purchasers only)
purchasers <- subset(northpoint, Purchase == 1)

#Merge all source columns into a single "Acquisition_Source"
source_cols <- c("source_a","source_b","source_c","source_d","source_e",
                 "source_m","source_o","source_h","source_r","source_s",
                 "source_t","source_u","source_p","source_x","source_w")

#Build a matrix of just the source flags
src_mat <- as.matrix(purchasers[, source_cols, drop = FALSE])

# For each row:
#if exactly one source == 1 -> use that source name
#if more than one == 1      -> "Multi"
#if none == 1               -> "Unknown"
pick_source <- function(x, coln) {
  idx <- which(x == 1)
  if (length(idx) == 1) return(coln[idx])
  if (length(idx) > 1)  return("Multi")
  return("Unknown")
}
purchasers$Acquisition_Source <- apply(src_mat, 1, pick_source, coln = colnames(src_mat))

#Short label (drop the "source_" prefix for nicer chart labels)
purchasers$AcqSrcShort <- ifelse(grepl("^source_", purchasers$Acquisition_Source),
                                 sub("^source_", "", purchasers$Acquisition_Source),
                                 purchasers$Acquisition_Source)

#(A) Spending by Acquisition Source (Top 6 by count)
src_counts <- sort(table(purchasers$AcqSrcShort), decreasing = TRUE)
top_src_names <- names(src_counts)[1:min(6, length(src_counts))]
top_idx <- purchasers$AcqSrcShort %in% top_src_names

#Boxplot
par(mar = c(5, 5, 4, 2))
boxplot(Spending ~ AcqSrcShort, data = purchasers[top_idx, ],
        main = "Spending by Acquisition Source (Top 6)",
        xlab = "Acquisition Source", ylab = "Spending ($)",
        las = 2, col = "lightgray")

#Add counts below x-axis labels for context
mtext(paste0("Counts: ", paste(src_counts[top_src_names], collapse = "  |  ")),
      side = 1, line = 5, cex = 0.8)

#(B) Spending by Web Order (Yes/No)
# Make a readable label
purchasers$WebOrderYN <- ifelse(purchasers$Web.order == 1, "Yes", "No")

par(mar = c(5, 5, 4, 2))
boxplot(Spending ~ WebOrderYN, data = purchasers,
        main = "Spending by Prior Web Order",
        xlab = "Prior Web Order", ylab = "Spending ($)",
        col = c("lightblue","lightgreen"))

#Show group counts on top
wo_counts <- table(purchasers$WebOrderYN)
text(x = c(1, 2), y = tapply(purchasers$Spending, purchasers$WebOrderYN, max, na.rm = TRUE) * 0.95,
     labels = paste("n=", wo_counts), cex = 0.9)

#(C) Spending vs Past-Year Purchase Frequency
par(mar = c(5, 5, 4, 2))
plot(purchasers$Freq, purchasers$Spending,
     pch = 16, col = rgb(0,0,0,0.5),
     main = "Spending vs. Past-Year Purchase Frequency",
     xlab = "Purchase Frequency (last year)", ylab = "Spending ($)")
#Add a simple LOWESS smoother
lw <- lowess(purchasers$Freq, purchasers$Spending, f = 2/3)
lines(lw, lwd = 2, col = "red")

#(D) Optional: Quick distribution view for Spending among purchasers
par(mar = c(5, 5, 4, 2))
hist(purchasers$Spending, breaks = 30, col = "lightcoral",
     main = "Purchasers: Spending Distribution", xlab = "Spending ($)")
mtext(paste("Median:", round(median(purchasers$Spending, na.rm = TRUE), 2)), side = 3)


##STEP 4B: Predictor Analysis for Classification
#Goal: Explore variables that help predict Purchase (0/1)
#Target variable: Purchase (0 = No, 1 = Yes)
table(northpoint$Purchase)

#1.Past Purchase Frequency vs Purchase (Boxplot)
boxplot(Freq ~ Purchase, data = northpoint,
        names = c("Non-Purchasers", "Purchasers"),
        main = "Frequency vs Purchase",
        xlab = "Purchase Status", ylab = "Past Purchase Frequency",
        col = c("tomato", "lightgreen"))
abline(h = median(northpoint$Freq), col = "blue", lty = 2)

#Web Order History vs Purchase (Bar plot)
web_counts <- table(northpoint$Web.order, northpoint$Purchase)
barplot(web_counts, beside = TRUE,
        col = c("lightblue","lightgreen"),
        legend.text = c("No Purchase","Purchase"),
        names.arg = c("No Web Order","Web Order"),
        main = "Web Order History vs Purchase",
        xlab = "Prior Web Order", ylab = "Count")
#Add labels
text(x = c(1.5,3.5,5.5,7.5), y = as.vector(web_counts)+10,
     labels = as.vector(web_counts), cex = 0.8)

#US vs Purchase (Bar plot)
us_counts <- table(northpoint$US, northpoint$Purchase)
barplot(us_counts, beside = TRUE,
        col = c("lightblue","lightgreen"),
        legend.text = c("No Purchase","Purchase"),
        names.arg = c("Non-US","US"),
        main = "US vs Purchase",
        xlab = "Geographic Flag", ylab = "Count")
text(x = c(1.5,3.5,5.5,7.5), y = as.vector(us_counts)+10,
     labels = as.vector(us_counts), cex = 0.8)

#Residential Address vs Purchase
addr_counts <- table(northpoint$Address_is_res, northpoint$Purchase)
barplot(addr_counts, beside = TRUE,
        col = c("gold","darkorange"),
        legend.text = c("No Purchase","Purchase"),
        names.arg = c("Non-Residential","Residential"),
        main = "Address Type vs Purchase",
        xlab = "Address Type", ylab = "Count")
text(x = c(1.5,3.5,5.5,7.5), y = as.vector(addr_counts)+10,
     labels = as.vector(addr_counts), cex = 0.8)

#Gender vs Purchase
gender_counts <- table(northpoint$Gender.male, northpoint$Purchase)
barplot(gender_counts, beside = TRUE,
        col = c("pink","lightgreen"),
        legend.text = c("No Purchase","Purchase"),
        names.arg = c("Other","Male"),
        main = "Gender vs Purchase",
        xlab = "Gender", ylab = "Count")
text(x = c(1.5,3.5,5.5,7.5), y = as.vector(gender_counts)+10,
     labels = as.vector(gender_counts), cex = 0.8)

#Acquisition Source vs Purchase (Top 6 sources)
source_cols <- grep("^source_", names(northpoint), value = TRUE)
#Count how many purchasers came from each source
src_purchase <- colSums(northpoint[northpoint$Purchase == 1, source_cols])
src_nonpurchase <- colSums(northpoint[northpoint$Purchase == 0, source_cols])

src_df <- data.frame(Source = sub("^source_", "", source_cols),
                     Purchasers = src_purchase,
                     NonPurchasers = src_nonpurchase)

#Sort by purchasers count
src_df <- src_df[order(-src_df$Purchasers), ]
top_src <- head(src_df, 6)

barplot(t(as.matrix(top_src[, c("NonPurchasers","Purchasers")])),
        beside = TRUE, col = c("tomato","lightgreen"),
        legend.text = c("Non-Purchasers","Purchasers"),
        names.arg = top_src$Source,
        main = "Top Acquisition Sources vs Purchase",
        xlab = "Source", ylab = "Count", las = 2)

#Feature Importance (basic Logistic Regression)
#Convert binary cols to factors for glm
northpoint$US <- as.factor(northpoint$US)
northpoint$Gender.male <- as.factor(northpoint$Gender.male)
northpoint$Web.order <- as.factor(northpoint$Web.order)
northpoint$Address_is_res <- as.factor(northpoint$Address_is_res)

logit_model <- glm(Purchase ~ Freq + US + Gender.male + Web.order + Address_is_res,
                   data = northpoint, family = binomial)

summary(logit_model)


#Step4C: Predictor Analysis for Regression
#Target: Spending (only for purchasers)
purchasers <- subset(northpoint, Purchase == 1)

#Correlation Matrix — Numeric Predictors
num_vars <- c("Spending", "Freq", "last_update_days_ago", "X1st_update_days_ago")
corr_matrix <- round(cor(purchasers[, num_vars], use = "complete.obs"), 2)
print("Correlation Matrix:")
print(corr_matrix)

#Optional: nice visualization (base R)
corrplot <- function(m) {
  op <- par(mar = c(1,1,4,1))
  image(1:ncol(m), 1:nrow(m), t(m[nrow(m):1, ]), col = heat.colors(15),
        axes = FALSE, main = "Correlation Matrix (Heatmap)")
  axis(1, at = 1:ncol(m), labels = colnames(m), las = 2)
  axis(2, at = 1:nrow(m), labels = rev(rownames(m)), las = 2)
  for (i in 1:nrow(m)) {
    for (j in 1:ncol(m)) {
      text(j, nrow(m)-i+1, labels = sprintf("%.2f", m[i,j]), cex = 0.8)
    }
  }
  par(op)
}
corrplot(corr_matrix)

#Linear Regression Model
lm_model <- lm(Spending ~ Freq + Web.order + US + Gender.male +
                 Address_is_res + last_update_days_ago + X1st_update_days_ago,
               data = purchasers)
summary(lm_model)


#Step 7: Data Partitioning (70/30)
set.seed(123) #reproducibility

#Classification split
pos_idx <- which(northpoint$Purchase == 1)
neg_idx <- which(northpoint$Purchase == 0)

train_pos_idx <- sample(pos_idx, size = floor(0.7 * length(pos_idx)))
train_neg_idx <- sample(neg_idx, size = floor(0.7 * length(neg_idx)))

train_idx_cls <- sort(c(train_pos_idx, train_neg_idx))
test_idx_cls  <- setdiff(seq_len(nrow(northpoint)), train_idx_cls)

train_cls <- northpoint[train_idx_cls, ]
test_cls  <- northpoint[test_idx_cls, ]

#Check proportions and sizes
print("Original Purchase proportion:")
print(prop.table(table(northpoint$Purchase)))

print("Train Purchase proportion:")
print(prop.table(table(train_cls$Purchase)))

print("Test Purchase proportion:")
print(prop.table(table(test_cls$Purchase)))

print(paste("Classification - Train size:", nrow(train_cls)))
print(paste("Classification - Test size:", nrow(test_cls)))


#Regression split (purchasers only)
purchasers <- subset(northpoint, Purchase == 1)
n_purch <- nrow(purchasers)

train_idx_reg <- sample(seq_len(n_purch), size = floor(0.7 * n_purch))
test_idx_reg  <- setdiff(seq_len(n_purch), train_idx_reg)

train_reg <- purchasers[train_idx_reg, ]
test_reg  <- purchasers[test_idx_reg, ]

#Check sizes
print(paste("Regression - Train size:", nrow(train_reg)))
print(paste("Regression - Test size:", nrow(test_reg)))



###Model Fitting, Validation and Test Accuracy
#Logistic Regression: M1 & M2
features_m1 <- c("Freq", "last_update_days_ago", "X1st_update_days_ago",
                 "Web.order", "Gender.male", "US", "Address_is_res")
features_m2 <- c(features_m1, source_cols) #source_cols is the cumulative column of all sources that I created earlier week

form_m1 <- as.formula(paste("Purchase ~", paste(features_m1, collapse = " + ")))
form_m2 <- as.formula(paste("Purchase ~", paste(features_m2, collapse = " + ")))

model1 <- glm(form_m1, data = train_cls, family = binomial())
model2 <- glm(form_m2, data = train_cls, family = binomial())

#PREDICTIONS (TEST)
probs_m1 <- predict(model1, newdata = test_cls, type = "response")
probs_m2 <- predict(model2, newdata = test_cls, type = "response")

#After probs_m1 / probs_m2 are computed
scale_factor <- 0.106  # = 5.3 / 50
probs_m1_cal <- pmin(pmax(probs_m1 * scale_factor, 0), 1)
probs_m2_cal <- pmin(pmax(probs_m2 * scale_factor, 0), 1)

pred_m1_cal <- ifelse(probs_m1_cal >= 0.053, 1, 0)  # threshold 0.053
pred_m2_cal <- ifelse(probs_m2_cal >= 0.053, 1, 0)

#METRICS
cm_m1 <- confusionMatrix(factor(pred_m1_cal, levels = c(0,1)),
                         factor(test_cls$Purchase, levels = c(0,1)))
cm_m2 <- confusionMatrix(factor(pred_m2_cal, levels = c(0,1)),
                         factor(test_cls$Purchase, levels = c(0,1)))

roc_m1 <- roc(test_cls$Purchase, probs_m1)
roc_m2 <- roc(test_cls$Purchase, probs_m2)
auc_m1 <- as.numeric(auc(roc_m1))
auc_m2 <- as.numeric(auc(roc_m2))

top_k_capture <- function(y_true, probs, k = 0.10) {
  n <- length(probs); top_n <- ceiling(n * k)
  ord <- order(probs, decreasing = TRUE)
  top_idx <- ord[1:top_n]
  captured <- sum(y_true[top_idx] == 1)
  total_pos <- sum(y_true == 1)
  if (total_pos == 0) return(NA_real_)
  captured / total_pos
}
m1_top10 <- top_k_capture(test_cls$Purchase, probs_m1, 0.10)
m2_top10 <- top_k_capture(test_cls$Purchase, probs_m2, 0.10)

# 1) ROC Curve: Model 1 vs Model 2
plot(roc_m1, col = "blue", lwd = 2, legacy.axes = TRUE,
     main = "ROC Curve: Model 1 vs Model 2")
plot(roc_m2, col = "red",  lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste0("Model 1 (AUC = ", sprintf("%.3f", auc_m1), ")"),
                  paste0("Model 2 (AUC = ", sprintf("%.3f", auc_m2), ")")),
       col = c("blue","red"), lwd = 2, bty = "n")

# 2) Cumulative Gains: Model 1 vs Model 2
make_gains <- function(actual, prob, label){
  df <- data.frame(actual = actual, prob = prob) %>% arrange(desc(prob))
  n <- nrow(df)
  df$rank <- seq_len(n)
  # Deciles by rank (10 bins with similar counts)
  br <- quantile(df$rank, probs = seq(0,1,0.1), type = 1)
  # Avoid duplicate breaks by jittering slightly if needed
  br <- unique(br)
  if (length(br) < 11) {
    br <- unique(round(seq(1, n, length.out = 11)))
  }
  df$decile <- cut(df$rank, breaks = br, include.lowest = TRUE, labels = FALSE)
  
  gains <- df %>%
    group_by(decile) %>%
    summarise(buyers = sum(actual), n = n(), .groups = "drop") %>%
    mutate(cum_buyers = cumsum(buyers),
           cum_pop = cumsum(n),
           cum_pct_buyers = cum_buyers / sum(buyers),
           cum_pct_pop = cum_pop / sum(n),
           model = label)
  gains
}

g1 <- make_gains(test_cls$Purchase, probs_m1, "Model 1")
g2 <- make_gains(test_cls$Purchase, probs_m2, "Model 2")
gains_df <- bind_rows(g1, g2)

ggplot(gains_df, aes(x = cum_pct_pop, y = cum_pct_buyers, color = model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(labels = percent, breaks = seq(0,1,0.1)) +
  scale_y_continuous(labels = percent, breaks = seq(0,1,0.1)) +
  labs(title = "Cumulative Gains: Model 1 vs Model 2",
       x = "Cumulative % of Population (by Score)",
       y = "Cumulative % of Buyers Captured") +
  theme_minimal()




# Random Forest (Classification)
#train_cls, test_cls already created; Acquisition_Source exists

#target & predictors: Keeping numeric copies of target for AUC / gains
y_train_num <- if (is.factor(train_cls$Purchase)) as.integer(as.character(train_cls$Purchase)) else train_cls$Purchase
y_test_num  <- if (is.factor(test_cls$Purchase))  as.integer(as.character(test_cls$Purchase))  else test_cls$Purchase

#RandomForest wants factor(target) for classification
train_cls$Purchase <- factor(y_train_num, levels = c(0,1))
test_cls$Purchase  <- factor(y_test_num,  levels = c(0,1))

#Feature set (aligned with Logistic Model 2 approach)
features <- c("Freq","last_update_days_ago","X1st_update_days_ago",
              "Web.order","Gender.male","US","Address_is_res",
              "Acquisition_Source")
features <- features[features %in% names(train_cls)]

#Business metric: Top 10% capture
top10_capture <- function(y_true_num, y_prob) {
  n <- length(y_prob); k <- ceiling(0.10 * n)
  ord <- order(y_prob, decreasing = TRUE)
  top_idx <- ord[1:k]
  captured <- sum(y_true_num[top_idx] == 1)
  total_pos <- sum(y_true_num == 1)
  if (total_pos == 0) return(NA_real_)
  captured / total_pos
}

# 1) Train 3 Random Forest candidates
p <- length(features)
mtry_default <- max(1, floor(sqrt(p)))

set.seed(42)
rf1 <- randomForest(
  x = train_cls[, features],
  y = train_cls$Purchase,
  ntree = 300,
  mtry = mtry_default,
  nodesize = 1,
  importance = TRUE
)

set.seed(42)
rf2 <- randomForest(
  x = train_cls[, features],
  y = train_cls$Purchase,
  ntree = 500,
  mtry = max(1, floor(p/3)),
  nodesize = 3,
  importance = TRUE
)

set.seed(42)
rf3 <- randomForest(
  x = train_cls[, features],
  y = train_cls$Purchase,
  ntree = 700,
  mtry = max(1, floor(p/2)),
  nodesize = 5,
  importance = TRUE
)


# 2) Evaluate on TEST set
# Probabilities of class "1"
prob1 <- predict(rf1, newdata = test_cls[, features], type = "prob")[, "1"]
prob2 <- predict(rf2, newdata = test_cls[, features], type = "prob")[, "1"]
prob3 <- predict(rf3, newdata = test_cls[, features], type = "prob")[, "1"]

#scaling to match real world
scale_factor <- 0.106
prob1_cal <- pmin(pmax(prob1 * scale_factor, 0), 1)
prob2_cal <- pmin(pmax(prob2 * scale_factor, 0), 1)
prob3_cal <- pmin(pmax(prob3 * scale_factor, 0), 1)

#predicted labels at 0.053 cutoff
pred1 <- ifelse(prob1_cal >= 0.053, 1, 0)
pred2 <- ifelse(prob2_cal >= 0.053, 1, 0)
pred3 <- ifelse(prob3_cal >= 0.053, 1, 0)

#caret confusion matrices
cm1 <- confusionMatrix(factor(pred1, levels = c(0,1)), factor(y_test_num, levels = c(0,1)))
cm2 <- confusionMatrix(factor(pred2, levels = c(0,1)), factor(y_test_num, levels = c(0,1)))
cm3 <- confusionMatrix(factor(pred3, levels = c(0,1)), factor(y_test_num, levels = c(0,1)))

#AUCs
roc1 <- roc(y_test_num, prob1); auc1 <- as.numeric(auc(roc1))
roc2 <- roc(y_test_num, prob2); auc2 <- as.numeric(auc(roc2))
roc3 <- roc(y_test_num, prob3); auc3 <- as.numeric(auc(roc3))

#Top-10% capture
t10_1 <- top10_capture(y_test_num, prob1)
t10_2 <- top10_capture(y_test_num, prob2)
t10_3 <- top10_capture(y_test_num, prob3)

#Summary table
get_row <- function(label, cm, auc_val, t10_val) {
  data.frame(
    Model = label,
    Accuracy    = round(as.numeric(cm$overall["Accuracy"]), 4),
    Sensitivity = round(as.numeric(cm$byClass["Sensitivity"]), 4),
    Specificity = round(as.numeric(cm$byClass["Specificity"]), 4),
    AUC         = round(auc_val, 3),
    Top10       = round(100 * t10_val, 1),
    row.names = NULL
  )
}
rf_results <- rbind(
  get_row("RF-1 (ntree=300, mtry=sqrt(p), nodesize=1)", cm1, auc1, t10_1),
  get_row("RF-2 (ntree=500, mtry=p/3, nodesize=3)",     cm2, auc2, t10_2),
  get_row("RF-3 (ntree=700, mtry=p/2, nodesize=5)",     cm3, auc3, t10_3)
)
print(rf_results, row.names = FALSE)


# 3) ROC: all 3 models on one plot (choose visually)
plot(roc1, col = "blue", lwd = 2, legacy.axes = TRUE,
     main = "ROC Curves: Random Forest Models (Test Set)")
plot(roc2, col = "red",        lwd = 2, add = TRUE)
plot(roc3, col = "darkgreen",  lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste0("RF-1 (AUC=", round(auc1, 3), ")"),
         paste0("RF-2 (AUC=", round(auc2, 3), ")"),
         paste0("RF-3 (AUC=", round(auc3, 3), ")")
       ),
       col = c("blue","red","darkgreen"), lwd = 2, bty = "n")


# 4) Pick BEST by AUC (tie-breaker: Accuracy)
# Business metric: Top 10% capture
# LIFT CHART: Compare RF-1, RF-2, RF-3
lift_tbl <- function(actual, prob, groups=10) {
  ord <- order(prob, decreasing = TRUE)
  a   <- actual[ord]
  n   <- length(a)
  pos <- sum(a)
  base_rate <- pos / n
  idx <- ceiling(seq(n/groups, n, by = n/groups))
  cum_buyers <- cumsum(a)[idx]
  cum_pop    <- idx
  cum_resp   <- cum_buyers / cum_pop
  cum_lift   <- cum_resp / base_rate
  data.frame(Cum_Pop_Pct = cum_pop / n, Cum_Lift = cum_lift)
}

lift1 <- lift_tbl(y_test_num, prob1)
lift2 <- lift_tbl(y_test_num, prob2)
lift3 <- lift_tbl(y_test_num, prob3)

plot(lift1$Cum_Pop_Pct, lift1$Cum_Lift, type="l", lwd=2, col="blue",
     xlab="Cumulative % of Population (by score)",
     ylab="Cumulative Lift (vs. overall rate)",
     main="Lift Chart (Cumulative): Random Forest Models",
     xaxs="i", yaxs="i")
lines(lift2$Cum_Pop_Pct, lift2$Cum_Lift, lwd=2, col="red")
lines(lift3$Cum_Pop_Pct, lift3$Cum_Lift, lwd=2, col="darkgreen")
abline(h=1, lty=2)
axis(1, at = seq(0,1,0.1), labels = paste0(seq(0,100,10), "%"))
legend("topright",
       legend = c("RF-1","RF-2","RF-3","Random lift=1"),
       col = c("blue","red","darkgreen","black"),
       lwd = c(2,2,2,1),
       lty = c(1,1,1,2),
       bty = "n")


# STEP 13: CLUSTERING ANALYSIS
#Filter only purchasers
purchasers <- subset(northpoint, Purchase == 1)

#Select numeric variables for clustering
cluster_vars <- c("Freq", "Spending", "last_update_days_ago", "X1st_update_days_ago")
cluster_data <- purchasers[, cluster_vars]

#Scale the data (very important for K-Means)
cluster_data_scaled <- scale(cluster_data)

#Silhouette Method — extra validation
sil_scores <- numeric(10)
for (k in 2:10) {
  km_model <- kmeans(cluster_data_scaled, centers = k, nstart = 25)
  ss <- silhouette(km_model$cluster, dist(cluster_data_scaled))
  sil_scores[k] <- mean(ss[, 3])
}

plot(2:10, sil_scores[2:10], type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Average Silhouette Width",
     main = "Silhouette Method for Optimal K")
abline(v = which.max(sil_scores), col = "blue", lty = 2)

best_k <- which.max(sil_scores)
print(paste("Best number of clusters suggested by Silhouette:", best_k))

#Final K-Means Clustering with best_k
set.seed(123)
final_kmeans <- kmeans(cluster_data_scaled, centers = best_k, nstart = 25)

#Add cluster labels back to purchaser data
purchasers$Cluster <- factor(final_kmeans$cluster)

#Profile Clusters
cluster_profile <- aggregate(cluster_data, by = list(Cluster = purchasers$Cluster), FUN = mean)
cluster_profile$Size <- as.numeric(table(purchasers$Cluster))
print(cluster_profile)

#Visualization (Simple Scatterplot)
#Example: Spending vs Freq colored by cluster
ggplot(purchasers, aes(x = Freq, y = Spending, color = Cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Customer Clusters: Spending vs Purchase Frequency") +
  theme_minimal()


# STEP 13: CLUSTERING (K-Means): Silhouette-only
# 1) Use purchasers only (clustering spending behavior makes sense for responders)
purchasers <- subset(northpoint, Purchase == 1)

# 2) Select continuous variables for clustering
cluster_vars  <- c("Freq", "Spending", "last_update_days_ago", "X1st_update_days_ago")
cluster_data  <- purchasers[, cluster_vars]

# 3) Scale features (essential for distance-based methods like K-Means)
cluster_data_scaled <- scale(cluster_data)

# 4) Choose K using average Silhouette Width (no elbow)
suppressPackageStartupMessages(library(cluster))
k_min <- 2; k_max <- 8
sil_scores <- rep(NA_real_, k_max)

set.seed(123)
for (k in k_min:k_max) {
  km_tmp <- kmeans(cluster_data_scaled, centers = k, nstart = 25)
  sil_tmp <- silhouette(km_tmp$cluster, dist(cluster_data_scaled))
  sil_scores[k] <- mean(sil_tmp[, 3])
}
best_k <- which.max(sil_scores)

# 5) Final K-Means with chosen K
set.seed(123)
final_kmeans <- kmeans(cluster_data_scaled, centers = best_k, nstart = 25)
purchasers$Cluster <- factor(final_kmeans$cluster)

# 6) Robust Silhouette Plot (explicit distance + graphics reset)
d <- dist(cluster_data_scaled, method = "euclidean")
sil_best <- silhouette(final_kmeans$cluster, d)

graphics.off(); par(mfrow = c(1,1)); par(mar = c(5,4,4,2) + 0.1); palette("default")
plot(sil_best,
     do.col.sort = TRUE,
     border = NA,
     col = rep(1:best_k, times = table(final_kmeans$cluster)),
     main = paste("Silhouette Plot (K =", best_k, ")"),
     xlab = "Silhouette width s(i)")

# 7) Cluster Profile (means in original units + cluster sizes) — ready for report
cluster_profile <- aggregate(cluster_data, by = list(Cluster = purchasers$Cluster), FUN = mean)
cluster_profile$Size <- as.numeric(table(purchasers$Cluster)[as.character(cluster_profile$Cluster)])
cluster_profile_report <- within(cluster_profile, {
  Freq <- round(Freq, 2)
  Spending <- round(Spending, 2)
  last_update_days_ago <- round(last_update_days_ago, 1)
  X1st_update_days_ago <- round(X1st_update_days_ago, 1)
})
cluster_profile_report  # view/print this table when you want

# 8) Simple visualization (Spending vs Frequency) colored by cluster
suppressPackageStartupMessages(library(ggplot2))
ggplot(purchasers, aes(x = Freq, y = Spending, color = Cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = paste("Customer Clusters (K =", best_k, "): Spending vs Frequency"),
       x = "Purchase Frequency", y = "Spending ($)") +
  theme_minimal()


# HIERARCHICAL CLUSTERING (Visual K selection)
# 1) Data subset (same vars as K-Means; purchasers only)
purchasers <- subset(northpoint, Purchase == 1)
hc_vars <- c("Freq", "Spending", "last_update_days_ago", "X1st_update_days_ago")
hc_data <- purchasers[, hc_vars]

# 2) Scale features (essential for distance-based clustering)
hc_scaled <- scale(hc_data)

# 3) Distance + Hierarchical clustering (Ward.D2 creates compact clusters)
d  <- dist(hc_scaled, method = "euclidean")
hc <- hclust(d, method = "ward.D2")

# 4) Dendrogram — visually inspect and decide K
graphics.off(); par(mfrow = c(1,1)); par(mar = c(4,4,2,2) + 0.1)
plot(hc, labels = FALSE, hang = -1, main = "Dendrogram (Ward.D2)",
     xlab = "Customers", ylab = "Height (Dissimilarity)")

#After inspecting the dendrogram, K is set as 3 and run the rest
K <- 3

# 5) Assign cluster labels
hclust_labels <- cutree(hc, k = K)
purchasers$HCluster <- factor(hclust_labels)

# 6) Silhouette validation (robust plotting)
suppressPackageStartupMessages(library(cluster))
sil_hc <- silhouette(hclust_labels, d)

graphics.off(); par(mfrow = c(1,1)); par(mar = c(5,4,4,2) + 0.1)
plot(sil_hc,
     do.col.sort = TRUE,
     border = NA,
     col = rep(1:K, times = table(hclust_labels)),
     main = paste("Silhouette Plot (Hierarchical, K =", K, ")"),
     xlab = "Silhouette width s(i)")
abline(v = mean(sil_hc[, "sil_width"]), lty = 2, col = "gray40")  # avg silhouette marker

# 7) Cluster profile (means in original units + size) ready for report
hc_profile <- aggregate(hc_data, by = list(Cluster = purchasers$HCluster), FUN = mean)
hc_profile$Size <- as.numeric(table(purchasers$HCluster)[as.character(hc_profile$Cluster)])
hc_profile_report <- within(hc_profile, {
  Freq <- round(Freq, 2)
  Spending <- round(Spending, 2)
  last_update_days_ago <- round(last_update_days_ago, 1)
  X1st_update_days_ago <- round(X1st_update_days_ago, 1)
})
hc_profile_report

# 8) Simple visualization: Spending vs Frequency colored by hierarchical clusters
suppressPackageStartupMessages(library(ggplot2))
ggplot(purchasers, aes(x = Freq, y = Spending, color = HCluster)) +
  geom_point(alpha = 0.7) +
  labs(title = paste("Hierarchical Clusters (K =", K, "): Spending vs Frequency"),
       x = "Purchase Frequency", y = "Spending ($)") +
  theme_minimal()



# REGRESSION COMPARISON
# Target: Spending (purchasers only)
# Models: LM-1, LM-2   |   RF-1, RF-2, RF-3
# Metrics: RMSE, MAE, R2 on TEST
# Plots: Actual vs Predicted for best Linear & best RF; RF importance

#Ensure train/test for purchasers exist
if (!exists("train_reg") | !exists("test_reg")) {
  purchasers <- subset(northpoint, Purchase == 1)
  set.seed(123)
  n_purch <- nrow(purchasers)
  train_idx_reg <- sample(seq_len(n_purch), size = floor(0.7 * n_purch))
  test_idx_reg  <- setdiff(seq_len(n_purch), train_idx_reg)
  train_reg <- purchasers[train_idx_reg, ]
  test_reg  <- purchasers[test_idx_reg, ]
}

# 1) Prep: factors, predictors
fac_cols <- c("Web.order","US","Gender.male","Address_is_res")
for (cc in fac_cols) {
  if (cc %in% names(train_reg)) train_reg[[cc]] <- as.factor(train_reg[[cc]])
  if (cc %in% names(test_reg))  test_reg[[cc]]  <- as.factor(test_reg[[cc]])
}

source_flags <- grep("^source_", names(train_reg), value = TRUE)

drop_zerovar <- function(df, cols) {
  zv <- sapply(df[, cols, drop = FALSE], function(x) length(unique(x)) <= 1)
  cols[!zv]
}

rmse_ <- function(a, p) sqrt(mean((a - p)^2))
mae_  <- function(a, p) mean(abs(a - p))
r2_   <- function(a, p) 1 - sum((a - p)^2) / sum((a - mean(a))^2)

# 2) MULTIPLE LINEAR REGRESSION (2 models)
# LM-1: Core behavioral + simple demographics
lm1_vars <- c("Freq","last_update_days_ago","X1st_update_days_ago",
              "Web.order","US","Gender.male","Address_is_res")
lm1_vars <- lm1_vars[lm1_vars %in% names(train_reg)]
lm1_vars <- drop_zerovar(train_reg, lm1_vars)
form_lm1 <- as.formula(paste("Spending ~", paste(lm1_vars, collapse = " + ")))

# LM-2: LM-1 + all source_* flags
lm2_vars <- unique(c(lm1_vars, source_flags))
lm2_vars <- lm2_vars[lm2_vars %in% names(train_reg)]
lm2_vars <- drop_zerovar(train_reg, lm2_vars)
form_lm2 <- as.formula(paste("Spending ~", paste(lm2_vars, collapse = " + ")))

LM1 <- lm(form_lm1, data = train_reg)
LM2 <- lm(form_lm2, data = train_reg)

pred_LM1 <- pmax(predict(LM1, newdata = test_reg), 0)
pred_LM2 <- pmax(predict(LM2, newdata = test_reg), 0)

m_lm <- tibble(
  Model = c("LM-1: Core","LM-2: +Sources"),
  RMSE  = c(rmse_(test_reg$Spending, pred_LM1),
            rmse_(test_reg$Spending, pred_LM2)),
  MAE   = c(mae_(test_reg$Spending, pred_LM1),
            mae_(test_reg$Spending, pred_LM2)),
  R2    = c(r2_(test_reg$Spending, pred_LM1),
            r2_(test_reg$Spending, pred_LM2))
)

order_lm <- order(m_lm$RMSE, m_lm$MAE, -m_lm$R2)
best_lm  <- m_lm[order_lm[1], ]
best_lm_name <- best_lm$Model
pred_best_lm <- if (best_lm_name == "LM-1: Core") pred_LM1 else pred_LM2
LM_best <- if (best_lm_name == "LM-1: Core") LM1 else LM2

# 3) RANDOM FOREST REGRESSION (3 models)
# Use same broad feature set as LM-2 (works well for RF)
rf_vars <- lm2_vars
rf_vars <- rf_vars[rf_vars %in% names(train_reg)]
rf_vars <- drop_zerovar(train_reg, rf_vars)
p <- length(rf_vars)

set.seed(42)
RF1 <- randomForest(x = train_reg[, rf_vars, drop = FALSE], y = train_reg$Spending,
                    ntree = 300, mtry = max(1, floor(sqrt(p))), nodesize = 5, importance = TRUE)
set.seed(42)
RF2 <- randomForest(x = train_reg[, rf_vars, drop = FALSE], y = train_reg$Spending,
                    ntree = 500, mtry = max(1, floor(p/3)), nodesize = 5, importance = TRUE)
set.seed(42)
RF3 <- randomForest(x = train_reg[, rf_vars, drop = FALSE], y = train_reg$Spending,
                    ntree = 700, mtry = max(1, floor(p/2)), nodesize = 3, importance = TRUE)

pred_RF1 <- predict(RF1, newdata = test_reg[, rf_vars, drop = FALSE])
pred_RF2 <- predict(RF2, newdata = test_reg[, rf_vars, drop = FALSE])
pred_RF3 <- predict(RF3, newdata = test_reg[, rf_vars, drop = FALSE])

m_rf <- tibble(
  Model = c("RF-1: 300/√p/ns=5","RF-2: 500/p/3/ns=5","RF-3: 700/p/2/ns=3"),
  RMSE  = c(rmse_(test_reg$Spending, pred_RF1),
            rmse_(test_reg$Spending, pred_RF2),
            rmse_(test_reg$Spending, pred_RF3)),
  MAE   = c(mae_(test_reg$Spending, pred_RF1),
            mae_(test_reg$Spending, pred_RF2),
            mae_(test_reg$Spending, pred_RF3)),
  R2    = c(r2_(test_reg$Spending, pred_RF1),
            r2_(test_reg$Spending, pred_RF2),
            r2_(test_reg$Spending, pred_RF3))
)

order_rf <- order(m_rf$RMSE, m_rf$MAE, -m_rf$R2)
best_rf  <- m_rf[order_rf[1], ]
best_rf_name <- best_rf$Model
pred_best_rf <- switch(best_rf_name,
                       "RF-1: 300/√p/ns=5" = pred_RF1,
                       "RF-2: 500/p/3/ns=5" = pred_RF2,
                       "RF-3: 700/p/2/ns=3" = pred_RF3)
RF_best <- switch(best_rf_name,
                  "RF-1: 300/√p/ns=5" = RF1,
                  "RF-2: 500/p/3/ns=5" = RF2,
                  "RF-3: 700/p/2/ns=3" = RF3)

# 4) Results & Plots
print(m_lm %>% mutate(across(c(RMSE,MAE), ~round(.,2)), R2 = round(R2,3)))
print(best_lm)
print(m_rf %>% mutate(across(c(RMSE,MAE), ~round(.,2)), R2 = round(R2,3)))
print(best_rf)

#Best Linear: Actual vs Predicted
ggplot(data.frame(Actual = test_reg$Spending, Pred = pred_best_lm),
       aes(x = Actual, y = Pred)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = paste("Best Linear:", best_lm_name, "| Actual vs Predicted"),
       x = "Actual Spending", y = "Predicted Spending") +
  theme_minimal()

#Best RF: Actual vs Predicted
ggplot(data.frame(Actual = test_reg$Spending, Pred = pred_best_rf),
       aes(x = Actual, y = Pred)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = paste("Best RF:", best_rf_name, "| Actual vs Predicted"),
       x = "Actual Spending", y = "Predicted Spending") +
  theme_minimal()

# RF Variable Importance
varImpPlot(RF_best, main = paste("Variable Importance —", best_rf_name))

# 5) Winner for Business Use
overall <- bind_rows(
  m_lm %>% filter(Model == best_lm_name) %>% mutate(Family = "Linear"),
  m_rf %>% filter(Model == best_rf_name) %>% mutate(Family = "Random Forest")
) %>% select(Family, Model, RMSE, MAE, R2)

print(overall %>% mutate(across(c(RMSE,MAE), ~round(.,2)), R2 = round(R2,3)))

pick_idx <- order(overall$RMSE, overall$MAE, -overall$R2)
overall_winner <- overall[pick_idx[1], ]
print(overall_winner)

