# Author: Jeremy Rubin
# Date: 7/27/25
# Code to do Random CLUSSO simulations based on real data

rm(list=ls())

library(readxl)
library(stringr)
library(glmnet)
library(mclust)
library(haven)
library(tictoc)
library(Matrix)
library(MixMatrix)

source("Random_CLUSSO_Functions_Simulate_Real_Data.R")
source("Mainfunction_albet_Flex_Thresh_2_20_24.r")
source("Mainfunction_albet_Structured_OLS.r")
source("mat_vec_prd.r")
source("coefficient.r")
source("K_prdu.r")

set.seed(87543)

#### PROCESS REAL DATA
#######################################################################################################################3
########### NEW CODE TO ACTUALLY RUN RANDOM CLUSSO, CLUSSO, AND THE NAIVE APPROACH ############################
### 2/20/24 NEW PARAMETERS TO DEFINE COEFFICIENT THRESHOLDS FOR 
### CLUSSO/FULL INFORMATION STRUCTURED LASSO AND RANDOM CLUSSO
random.thresh <- 0.01
clusso.thresh <- 0.01

B <- 200
# B <- 3

G <- 2

## Grid of regularization parameters for CLUSSO/Random CLUSSO
lambda.grid <- c(seq(0.1,1,0.1),seq(1.5,5,0.5))

################ PROCESS NEPTUNE DATA ###########################################
# setwd("C:/Users/jerem/Downloads/")
Tubules_PtVars <- read_sas("ptvars.sas7bdat")
tubule_features <- read_sas("tubulefeatures.sas7bdat")

# eGFRatBx is outcome from ptvars
# Both SAFID and BiopsyID for ptvars 

# tubule_features has BiopsyID 
all_tubule_data <- merge(Tubules_PtVars,tubule_features,by="BiopsyID",all.x=TRUE)
unique_SAFID <- unique(all_tubule_data$SAFID)

######## Get rid of any features which are min/max/std
features.remove <- str_detect(colnames(all_tubule_data),pattern="MAX") |
  str_detect(colnames(all_tubule_data),pattern="MIN") |
  str_detect(colnames(all_tubule_data),pattern="STD") |
  str_detect(colnames(all_tubule_data),pattern="STDEV")

all_tubule_reduced <- all_tubule_data[,!features.remove]

## Then, take out clinical/demographic features and other extraneous information
## like the file name 
all_tubule_reduced <- all_tubule_reduced[,48:113]

more.features.remove <- str_detect(colnames(all_tubule_reduced),pattern="15") |
  str_detect(colnames(all_tubule_reduced),pattern="20") |
  str_detect(colnames(all_tubule_reduced),pattern="30")

all_tubule_reduced <- all_tubule_reduced[,!more.features.remove]

### NEW CODE 6/7/24 TO REMOVE TWO FEATURES SUCH THAT X.AVG HAS NO PERFECTLY CORRELATED FEATURES
more.bad.features <- c(which(colnames(all_tubule_reduced)=="TE_LUMEN_TBM_AREA_RATIO"),
                       which(colnames(all_tubule_reduced)=="TE_TE_LUMEN_AREA_RATIO"),
                       
                       which(colnames(all_tubule_reduced)=="TE_LUMEN_AREA"),
                       which(colnames(all_tubule_reduced)=="TE_LUMEN_DIAMETER"),
                       which(colnames(all_tubule_reduced)=="LUMEN_TE_LUMEN_AREA_RATIO"),
                       which(colnames(all_tubule_reduced)=="TE_LUMEN_DIAMETER_TBM_THICK_RATI"),
                       which(colnames(all_tubule_reduced)=="LUMEN_DIAMETER_TE_LUMEN_DIAMETER"),
                       which(colnames(all_tubule_reduced)=="TBM_DIAMETER_TE_LUMEN_DIAMETER_R"),
                       which(colnames(all_tubule_reduced)=="NUCLEI_LUMEN_BORDER_DISTANCE_AVE"),
                       which(colnames(all_tubule_reduced)=="NUCLEI_TE_BORDER_CENT_AVE")
)
all_tubule_reduced <- all_tubule_reduced[,-more.bad.features]

tubule.colnames.save <- colnames(all_tubule_reduced)

eGFR.all <- all_tubule_data$eGFRatBx
SAFID.all <- all_tubule_data$SAFID

### Get rid of tubules that have missing values for features of interest
missing.mat <- apply(all_tubule_reduced,FUN=is.na,MARGIN=2)
row.missing.count <- rowSums(missing.mat)
paste("Number of tubules missing one or more features: ",sum(row.missing.count>0))
tubule.remove <- row.missing.count>0
all_tubule_reduced <- all_tubule_reduced[!tubule.remove,] 
eGFR.all <- eGFR.all[!tubule.remove]
SAFID.all <- SAFID.all[!tubule.remove]

X.list <- vector("list", length(unique_SAFID))
X.avg <- matrix(0,nrow=length(unique_SAFID),ncol=ncol(all_tubule_reduced))
eGFR <- rep(0,length(unique_SAFID))

for(i in 1:length(unique_SAFID))
{
  cur_pat <- unique_SAFID[i]
  
  cur_rows <- all_tubule_reduced[SAFID.all==cur_pat,]
  cur_eGFR <- eGFR.all[SAFID.all==cur_pat]
  eGFR[i] <- cur_eGFR[1]
  
  X.list[[i]] <- cur_rows
  X.avg[i,] <- colMeans(X.list[[i]])
}

# Get subjects which don't have an eGFR recorded
no.eGFR <- which(is.na(eGFR))

paste("Number of subjects who don't have an EGFR: ",length(no.eGFR))

# Remove these subjects from your design matrices and outcome vector
eGFR.NEPTUNE <- eGFR[-no.eGFR]

X.avg.NEPTUNE <- X.avg[-no.eGFR,]

X.list.NEPTUNE <- X.list[-no.eGFR]

### Remove rownames from X.list.NEPTUNE
for(i in 1:length(X.list.NEPTUNE)){rownames(X.list.NEPTUNE[[i]]) <- NULL}

###################################################### PROCESS CureGN DATA ####

Tubules_PtVars <- read_sas("tubf_sumstat_ptf_ptvar.sas7bdat")
tubule_features <- read_sas("curegn_tubulefeatures.sas7bdat")

# eGFRatBx is outcome from ptvars
# Both BiopsyID and BiopsyID for ptvars 

# tubule_features has BiopsyID 
all_tubule_data <- merge(Tubules_PtVars,tubule_features,by="BiopsyID",all.x=TRUE)
unique_BiopsyID <- unique(all_tubule_data$BiopsyID)

eGFR.all <- all_tubule_data$eGFRatBx
BiopsyID.all <- all_tubule_data$BiopsyID

######## Get rid of any features which are min/max/std
features.remove <- str_detect(colnames(all_tubule_data),pattern="MAX") |
  str_detect(colnames(all_tubule_data),pattern="MIN") |
  str_detect(colnames(all_tubule_data),pattern="STD") |
  str_detect(colnames(all_tubule_data),pattern="STDEV")

all_tubule_reduced <- all_tubule_data[,!features.remove]

## Then, take out clinical/demographic features and other extraneous information
## like the file name 
all_tubule_reduced <- all_tubule_reduced[,59:124]

more.features.remove <- str_detect(colnames(all_tubule_reduced),pattern="15") |
  str_detect(colnames(all_tubule_reduced),pattern="20") |
  str_detect(colnames(all_tubule_reduced),pattern="30")

all_tubule_reduced <- all_tubule_reduced[,!more.features.remove]

### NEW CODE 6/7/24 TO REMOVE TWO FEATURES SUCH THAT X.AVG HAS NO PERFECTLY CORRELATED FEATURES
more.bad.features <- c(which(colnames(all_tubule_reduced)=="TE_LUMEN_TBM_AREA_RATIO"),
                       which(colnames(all_tubule_reduced)=="TE_TE_LUMEN_AREA_RATIO"),
                       
                       which(colnames(all_tubule_reduced)=="TE_LUMEN_AREA"),
                       which(colnames(all_tubule_reduced)=="TE_LUMEN_DIAMETER"),
                       which(colnames(all_tubule_reduced)=="LUMEN_TE_LUMEN_AREA_RATIO"),
                       which(colnames(all_tubule_reduced)=="TE_LUMEN_DIAMETER_TBM_THICK_RATI"),
                       which(colnames(all_tubule_reduced)=="LUMEN_DIAMETER_TE_LUMEN_DIAMETER"),
                       which(colnames(all_tubule_reduced)=="TBM_DIAMETER_TE_LUMEN_DIAMETER_R"),
                       which(colnames(all_tubule_reduced)=="NUCLEI_LUMEN_BORDER_DISTANCE_AVE"),
                       which(colnames(all_tubule_reduced)=="NUCLEI_TE_BORDER_CENT_AVE"))

all_tubule_reduced <- all_tubule_reduced[,-more.bad.features]

######## NEW CODE 5/28/24 TO CORRECT THE THREE FEATURES THAT SHOULD ACTUALLY BE THE RECIPROCAL ####
all_tubule_reduced$TE_LUMEN_DIAMETER_TUBULE_DIAMETE <- 1/(all_tubule_reduced$TE_LUMEN_DIAMETER_TUBULE_DIAMETE)
all_tubule_reduced$TBM_THICK_TUBULE_DIAMETER_RATIO <- 1/(all_tubule_reduced$TBM_THICK_TUBULE_DIAMETER_RATIO)
all_tubule_reduced$TBM_TUBULE_AREA_RATIO <- 1/(all_tubule_reduced$TBM_TUBULE_AREA_RATIO)

colnames(all_tubule_reduced) <- tubule.colnames.save

### Get rid of tubules that have missing values for features of interest
missing.mat <- apply(all_tubule_reduced,FUN=is.na,MARGIN=2)
row.missing.count <- rowSums(missing.mat)
paste("Number of tubules missing one or more features: ",sum(row.missing.count>0))
tubule.remove <- row.missing.count>0
all_tubule_reduced <- all_tubule_reduced[!tubule.remove,] 
eGFR.all <- eGFR.all[!tubule.remove]
BiopsyID.all <- BiopsyID.all[!tubule.remove]

X.list <- vector("list", length(unique_BiopsyID))
X.avg <- matrix(0,nrow=length(unique_BiopsyID),ncol=ncol(all_tubule_reduced))
eGFR <- rep(0,length(unique_BiopsyID))

for(i in 1:length(unique_BiopsyID))
{
  cur_pat <- unique_BiopsyID[i]
  
  cur_rows <- all_tubule_reduced[BiopsyID.all==cur_pat,]
  cur_eGFR <- eGFR.all[BiopsyID.all==cur_pat]
  eGFR[i] <- cur_eGFR[1]
  
  X.list[[i]] <- cur_rows
  X.avg[i,] <- colMeans(X.list[[i]])
}

# Get subjects which don't have an eGFR recorded
no.eGFR <- which(is.na(eGFR))

paste("Number of subjects who don't have an EGFR: ",length(no.eGFR))

# Remove these subjects from your design matrices and outcome vector
eGFR.CureGN <- eGFR[-no.eGFR]

X.avg.CureGN <- X.avg[-no.eGFR,]

X.list.CureGN <- X.list[-no.eGFR]

### Remove rownames from X.list.CureGN
for(i in 1:length(X.list.CureGN)){rownames(X.list.CureGN[[i]]) <- NULL}

############# COMBINE COHORTS ############################
eGFR.combined <- c(eGFR.NEPTUNE,eGFR.CureGN)
X.avg.combined <- rbind(X.avg.NEPTUNE,X.avg.CureGN)
X.list.combined <- c(X.list.NEPTUNE,X.list.CureGN)

# Histogram of the number of gloms per subject
tubule.count <- rep(0,length(X.list.combined))

for(i in 1:length(X.list.combined))
{
  tubule.count[i] <- dim(X.list.combined[[i]])[1]  
}

# Plot distribution of tubules
hist(tubule.count,main="",col="darkblue",xlab="")

####### Look at correlation matrix for X.avg ###### 

#### As a first pass for reducing correlation between features, remove all features that compute min/max/std 
colnames(X.avg.combined) <- tubule.colnames.save

n <- length(eGFR.combined)

#### END PROCESS REAL DATA
###########################################################################################################################
### Do training/testing split and clustering separately for training and testing data

set.seed(1092023)

##### PARAMETERS YOU'RE CHANGING FOR SIMULATIONS ############
## Sample size n, correlation rho, and number of features

### Need a new beta_star - can be 60% sparse but q is different
q <- dim(X.avg.combined)[2]

#########################################################################################

sigma_sq <- 1
# sigma_sq_m <- ceiling(IQR(tubule.count))
sigma_sq_m <- 5

# mu <- median(tubule.count)
mu <- 100

sigma.r <- 1

alpha_star <- sample(1:4,size=G,replace=TRUE)

############################################################################################ 
########### BEFORE ANY SIMULATIONS, DO CLUSSO ON TRUE PREDICTORS TO GENERATE REMAINING SIMULATED DATA
### Going to generate true weights based on doing CLUSSO to real data
w <- rep(0,n)

H <- floor(sqrt(12*sigma_sq_m+1))

# Make H odd
if(H %% 2 == 0){H <- H - 1}
a <- mu-(H-1)/2
b <- mu+(H-1)/2

G <- length(alpha_star)

# List of observed matrices 
raw.X <- X.list.combined

########################## New code to do clustering followed by ###########
#### Averaging by subpopulation before plugging into structured lasso 
# Then take row averages among clusters and regress on list of c x q 
# matrices, where c is the number of subpopulations 

X.all <- do.call(rbind, raw.X)

# do clustering on all data
NMM.out <- Mclust(X.all,G=G)$classification

#######################################################################################
# Making the cluster assignments from GMM clustering the second column of the observed
# Xi, and the true cluster labels the first column
#######################################################################################

for(i in 1:n)
{
  raw.X[[i]] <- cbind(NMM.out[1:nrow(raw.X[[i]])],raw.X[[i]])
  NMM.out <- NMM.out[(nrow(raw.X[[i]])+1):length(NMM.out)]
}

# List of c x q matrices, where c is the number of clusters per subject
# clust.X <- vector(mode='list', length=n)
clust.X <- array(1, dim=c(G,q,n))

### New code to save the true cluster-averaged feature matrices without being weighted
clust.X.only.avg <- array(1, dim=c(G,q,n))

for(i in 1:n)
{
  clust1.X <- raw.X[[i]][raw.X[[i]][,1]==1,]
  clust2.X <- raw.X[[i]][raw.X[[i]][,1]==2,]
  
  w1 <- nrow(clust1.X)/nrow(raw.X[[i]])
  w[i] <- w1
    
  clust1.avg <- apply(clust1.X[,2:ncol(clust1.X)],MARGIN=2,mean)
  clust2.avg <- apply(clust2.X[,2:ncol(clust2.X)],MARGIN=2,mean)
  
  clust.X[,,i] <- rbind(w1*clust1.avg,
                          (1-w1)*clust2.avg)
  
  clust.X.only.avg[,,i] <- rbind(clust1.avg,clust2.avg)
  
}

####################### New code 7/20/25 for beta_star where you pick features based stacking the cluster-level averages
## Want 40% of the features to be informative ` about 18 or 19`

# Step 1: Rearrange dimensions from (G, q, n) to (G, n, q)
# temp_array <- aperm(clust.X, c(1, 3, 2))
# 
# # Step 2: Reshape to a matrix of size (G*n, q)
# big_matrix <- matrix(temp_array, nrow = G * n, ncol = q)
# 
# # Step 3: Calculate correlation matrix of the q features
# cor_matrix <- cor(big_matrix)
# 
# # Calculate number of informative features (40% of total)
# n_informative <- floor(0.4 * q)
# cat("Total features:", q, "\n")
# cat("Selecting", n_informative, "informative features (40%)\n")
# 
# # Method 1: Select the most densely connected cluster
# # Convert correlation to distance
# # cor_dist <- as.dist(1 - abs(cor_matrix))
# # 
# # # Perform hierarchical clustering
# # hc <- hclust(cor_dist, method = "ward.D2")
# # 
# # # # Cut the tree to get clusters (adjust k based on dendrogram)
# # clusters <- cutree(hc, k = 4)  # try different values of k
# # # 
# # # # See which features are in each cluster
# # for(i in 1:max(clusters)) {
# #   cat("Cluster", i, ":", which(clusters == i), "\n")
# # }
# 
# # Method 2: Alternative approach - Select top correlated features
# # Calculate average absolute correlation for each feature with all others
# avg_cor <- apply(abs(cor_matrix), 1, function(x) mean(x[x < 1])) # exclude self-correlation
# 
# # Select features with highest average correlations
# top_features_method2 <- sort(order(avg_cor, decreasing = TRUE)[1:n_informative])

# 
# ### Candidate clusters based on hierarchical clustering cutting dendogram with k = 4
# ## 1, 8, 31
# ## 2, 4, 5
# 
# # Method 2: Find features with high within-group correlation
# # Set a threshold for "high correlation"
# threshold <- 0.7
# 
# # Function to find tight clusters
# find_tight_clusters <- function(cor_mat, min_cor = 0.7, min_size = 3) {
#   n <- nrow(cor_mat)
#   clusters <- list()
#   used <- rep(FALSE, n)
#   
#   for(i in 1:n) {
#     if(used[i]) next
#     
#     # Find features highly correlated with feature i
#     highly_corr <- which(abs(cor_mat[i, ]) >= min_cor)
#     
#     if(length(highly_corr) >= min_size) {
#       # Check if they're all highly correlated with each other
#       submat <- cor_mat[highly_corr, highly_corr]
#       if(min(abs(submat[upper.tri(submat)])) >= min_cor * 0.8) {
#         clusters[[length(clusters) + 1]] <- highly_corr
#         used[highly_corr] <- TRUE
#       }
#     }
#   }
#   
#   return(clusters)
# }
# 
# tight_clusters <- find_tight_clusters(cor_matrix, min_cor = 0.7, min_size = 3)
# 
# # Print the tight clusters
# for(i in seq_along(tight_clusters)) {
#   cat("Tight cluster", i, ":", tight_clusters[[i]], "\n")
# }
# 
# ## More clusters to consider
# ## 1, 8, 31
# ## 22, 27, 42
# 
# 
# # Method 3: Use network analysis to find communities
# library(igraph)
# 
# # Create adjacency matrix (correlations above threshold)
# adj_matrix <- abs(cor_matrix) > 0.7
# diag(adj_matrix) <- FALSE
# 
# # Create network
# g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected")
# 
# # Find communities
# communities <- cluster_walktrap(g)
# 
# # See the communities
# which(membership(communities)==2)

## Smallest community from network analysis is 1, 2, 31

## New code for beta_star 6/10/25 to hand-pick pathomic features based ones that have good block
## correlation with each other but not much correlation with other features
# "NUCLEI_TE_AREA_RATIO"       "NUCLEI_TE_LUMEN_AREA_RATIO" "NUCLEI_TUBULE_AREA_RATIO"
## These three features appear to be the most cleanly correlated with each other but not other features

inf_features <- 39:41

beta_star <- rep(0,q)
beta_star[inf_features] <- 15

s2 <- round(1-sum(beta_star!=0)/length(beta_star),digits=3)

########################################################################################3

#### New code to read in simulation parameter combination as command line argument
sim.num = as.numeric(commandArgs(TRUE)[[1]])

set.seed(sim.num)
print(paste("Simulation number: ",sim.num))

CLUSSO.stuff <- CLUSSO.performance(n,sigma_sq,alpha_star,beta_star,sigma.r,lambda.grid,clusso.thresh,random.thresh,clust.X,clust.X.only.avg,w)
  
CLUSSO.stats <- CLUSSO.stuff[[1]][,1]
random.CLUSSO.stats <- CLUSSO.stuff[[1]][,2]
Avg.stats <- CLUSSO.stuff[[1]][,3]
Full.stats <- CLUSSO.stuff[[1]][,4]

results.mat <- cbind(CLUSSO.stats,random.CLUSSO.stats,Avg.stats,Full.stats)  

rownames(results.mat) <- c("TPR","FPR","L1 norm of bias","MSE")
colnames(results.mat) <- c("CLUSSO","Random CLUSSO","Average balancing","Full information")

write.csv(results.mat,file=paste("Simulation_number_",sim.num,".csv",sep=""), row.names=FALSE)