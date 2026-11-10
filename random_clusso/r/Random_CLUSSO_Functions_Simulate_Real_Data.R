# Author: Jeremy Rubin
# Date: 6/1/24
# Random CLUSSO functions
# Functions to help run Random CLUSSO...

# Function to compare the performance of CLUSSO to the naive method and the Full Information Structured Lasso
CLUSSO.performance = function(n,sigma_sq,alpha_star,beta_star,
                              sigma.r,
                              lambda.grid,clusso.thresh,random.thresh,
                              clust.X,clust.X.only.avg,w)
{
  # Initialize beta coefficient vector
  beta_init <- rnorm(length(beta_star))
  
  CLUSSO.stuff <- generate.design.matrices(n,sigma_sq,alpha_star,beta_star,
                                           sigma.r,
                                           clust.X,clust.X.only.avg,w)

  Y <- CLUSSO.stuff[[1]]
  X.avg <- CLUSSO.stuff[[2]]
  simulated.X <- CLUSSO.stuff[[3]]
  clust.X.CLUSSO <- CLUSSO.stuff[[4]]
  time.to.avg <- CLUSSO.stuff[[5]]
  no.clust <- CLUSSO.stuff[[6]]
  
  ### While there are no tubules in a cluster for at least one subject, try simulating feature
  ### matrices and clustering again 
  while(no.clust==T)
  {
    print(paste("Trying to simulate data again!"))
    CLUSSO.stuff <- generate.design.matrices(n,sigma_sq,alpha_star,beta_star,
                                             sigma.r,
                                             clust.X,clust.X.only.avg,w)
    
    Y <- CLUSSO.stuff[[1]]
    X.avg <- CLUSSO.stuff[[2]]
    simulated.X <- CLUSSO.stuff[[3]]
    clust.X.CLUSSO <- CLUSSO.stuff[[4]]
    time.to.avg <- CLUSSO.stuff[[5]]
    no.clust <- CLUSSO.stuff[[6]]
  }
  
  print(paste("Done simulating data!"))
  
    ##### AVERAGE BALANCING RESULTS #########
    tic()
    
    avg.fit <- cv.glmnet(X.avg,Y,nfolds = 5)
    
    exectime <- toc()
    time.to.fit.naive <- exectime$toc - exectime$tic
    
    beta_coeff <- as.vector(coef(avg.fit, s = "lambda.min"))
    intercept <- beta_coeff[1]
    beta_hat.avg <- beta_coeff[-1]
    
    beta_hat.avg.original <- beta_hat.avg
    
    ### New code 3/28 to prevent normalizing beta_hat.avg vector if it is a vector 
    ### of all zeros
    # Check if the vector is not all zeros
    if (sum(beta_hat.avg == 0) != length(beta_hat.avg)) {
      beta_hat.avg <- beta_hat.avg/sum(abs(beta_hat.avg))
    } 
  
    zero.inds <- abs(beta_hat.avg)<clusso.thresh
    beta_hat.avg[abs(beta_hat.avg)<clusso.thresh]=0 
    beta_star <- beta_star/sum(abs(beta_star))
    
    TPR.avg <- (length(intersect(which(beta_hat.avg!=0),which(beta_star!=0)))/length(which(beta_star!=0)))*100
    FPR.avg <- (length(intersect(which(beta_hat.avg!=0),which(beta_star==0)))/length(which(beta_star==0)))*100
    bias.avg <- sum(abs(beta_star-beta_hat.avg))
  
    ## MSE
    beta_hat.avg.original[zero.inds]=0
    SSE <- 0
  
    for(r in 1:length(Y)){SSE <- SSE + (Y[r] - beta_hat.avg.original %*% X.avg[r,] - intercept)^2}
    MSE.avg <- SSE / length(Y)
    
    print(paste("Done with naive approach!"))
    
    ########### FULL INFORMATION RESULTS #################################################
    alpha_init <- rnorm(dim(clust.X)[1])
    
    tic()
    
    ############ CV-BASED LAMBDA ########################################################
    mse.full.lambda <- sapply(lambda.grid,lambda.CV.mse,X.train=clust.X,
                              Y.train=Y,
                              alpha_init=alpha_init,beta_init=beta_init,
                              thresh=clusso.thresh)
    lambda.full <- lambda.grid[which(mse.full.lambda==min(mse.full.lambda))][1]
    ####################################################################################
    
    slasso_results.full <- Mainfunction_albet(clust.X,Y,alpha_init,beta_init,lambda.full,clusso.thresh)
    
    exectime <- toc()
    time.to.fit.full <- exectime$toc - exectime$tic
    
    beta_hat.full <- slasso_results.full[[2]]
    alpha_hat.full <- slasso_results.full[[1]]
    
    simulation.stuff <- simulation.results(beta_hat.full,beta_star,alpha_hat.full,Y,clust.X)
    TPR.full <- simulation.stuff[1]
    FPR.full <- simulation.stuff[2]
    bias.full <- simulation.stuff[3]
    MSE.full <- simulation.stuff[4]
    
    
    print(paste("Done with Full Information Structured Lasso!"))
    
    ########### CLUSSO RESULTS #################################################
    alpha_init <- rnorm(dim(clust.X)[1])
    G <- length(alpha_init)
    
    tic()
    
    ############ CV-BASED LAMBDA ########################################################
    mse.clust.lambda <- sapply(lambda.grid,lambda.CV.mse,X.train=clust.X.CLUSSO,
                               Y.train=Y,
                               alpha_init=alpha_init,beta_init=beta_init,
                               thresh=clusso.thresh)
    lambda.clust <- lambda.grid[which(mse.clust.lambda==min(mse.clust.lambda))][1]
    ####################################################################################
    
    clust.fit <- Mainfunction_albet(clust.X.CLUSSO,Y,alpha_init,beta_init,lambda.clust,clusso.thresh)
    
    exectime <- toc()
    time.to.fit.CLUSSO <- exectime$toc - exectime$tic
    
    alpha_hat.clust <- clust.fit[[1]]  
    beta_hat.clust <- clust.fit[[2]]
    
    simulation.stuff <- simulation.results(beta_hat.clust,beta_star,alpha_hat.clust,Y,clust.X.CLUSSO)
    TPR.clust <- simulation.stuff[1]
    FPR.clust <- simulation.stuff[2]
    bias.clust <- simulation.stuff[3]
    MSE.clust <- simulation.stuff[4]
    
    
    print(paste("Done with CLUSSO!"))
    
    ########################## RANDOM CLUSSO RESULTS ##################################
    ###################### BOOTSTRAP ROUND 1 ##########################################
    ### Start with randomly drawing B bootstrap samples with replacement of size n 
    ## From original training set
    
    ### First alpha number of entries are the alpha coefficients 
    # Then, beta number of entries are the beta coefficients 
    # Then, a 0/1 indicator of whether there was an issue with clustering
    # Then, a single number for the MSE
    # -1 in the sample weights because we are passing in the true and estimated 
    # cluster labels
    # Another column for the lambda's picked by random CLUSSO for each bootstrap iteration
    # Hence + 3 
    # Extra three values that are returned: no.clust,lambda.clust
    
    tic()
    
    beta.matrix.plus.clust <- sapply(1:B,FUN=generate.importance.betas,raw.X=simulated.X,
                             Y=Y,G=G,lambda.grid=lambda.grid,
                             sample.weights=rep(0,ncol(simulated.X[[1]])-1),
                             thresh=random.thresh,simplify=FALSE)
    
    beta.matrix.reformat <- matrix(0,nrow=B,ncol=length(alpha_star)+length(beta_star)+2)
    for(i in 1:B){beta.matrix.reformat[i,] <- beta.matrix.plus.clust[[i]]}
    
    ###### Code to redo bootstrap repetitions that didn't work ############
    num.bad.clusters <- sum(beta.matrix.reformat[,ncol(beta.matrix.reformat)-1])
    while(num.bad.clusters>0)
    {
      print(paste("Number of bootstrap repetitions to redo: ",num.bad.clusters))
      beta.redo <- which(beta.matrix.reformat[,ncol(beta.matrix.reformat)-1]==1)
      beta.matrix.redo <- sapply(beta.redo,FUN=generate.importance.betas,raw.X=simulated.X,
                                       Y=Y,G=G,lambda.grid=lambda.grid,
                                 sample.weights=rep(0,ncol(raw.X[[1]])-1),
                                 thresh=random.thresh,
                                 simplify=FALSE)
      
      beta.matrix.redo.reformat <- matrix(0,nrow=length(beta.redo),ncol=length(alpha_star)+length(beta_star)+2)
      for(k in 1:length(beta.redo)){beta.matrix.redo.reformat[k,] <- beta.matrix.redo[[k]]}
      
      beta.matrix.reformat[beta.redo,] <- beta.matrix.redo.reformat
      num.bad.clusters <- sum(beta.matrix.reformat[,ncol(beta.matrix.reformat)-1])
    }
    
    ## Importance measure for each predictor is the absolute value of the average
    ## of bootstrapped coefficients
    
    ### Save random CLUSSO lambda values
    random.CLUSSO.r1.lambda <- beta.matrix.reformat[,ncol(beta.matrix.reformat)]
    
    importance.measures <- apply(beta.matrix.reformat,MARGIN=2,mean)
    importance.measures <- abs(importance.measures)
    
    ## The importance measures that you actually want are just the ones corresponding
    ## to the beta coefficients
    ## First alpha_star values are the estimate of alpha_star
    ## Last two columns are binary indicator of bootstrap repetitions 
    ## that didn't work and average lambda across bootstrap iterations
    beta.importance <- importance.measures[-c(1:length(alpha_star),
                                           (length(importance.measures)-1):length(importance.measures))]
    
    
    
    print(paste("Done with R1 bootstrapping for Random CLUSSO!"))
    
    ############################################################################################
    ### Then, do same procedure as before except use sampling weights proportional 
    ## to importance.measures
    ## Store beta coefficients from each bootstrap repetition
    ## Two extra columns: First extra column is for clustering accuracy
    ## and the second one keeps track of whether the clustering didn't work
    ## at least one subject didn't have any objects in one of the clusters
    ## One more column for random CLUSSO lambda
    
    ## Store clustering accuracy from each bootstrap repetition
    beta.matrix.final <- sapply(1:B,FUN=generate.importance.betas,raw.X=simulated.X,
                                     Y=Y,G=G,lambda.grid=lambda.grid,
                                     sample.weights=beta.importance,
                                     thresh=random.thresh,
                                     simplify=FALSE)
    
    beta.matrix.final.reformat <- matrix(0,nrow=B,ncol=length(alpha_star)+length(beta_star)+2)
    for(i in 1:B){beta.matrix.final.reformat[i,] <- beta.matrix.final[[i]]}
    
    ### Code to redo bootstrap iterations in second round of bootstrapping that did not 
    ### have valid clustering
    ###### Code to redo bootstrap repetitions that didn't work ############
    num.bad.clusters <- sum(beta.matrix.final.reformat[,ncol(beta.matrix.final.reformat)-1])
    while(num.bad.clusters>0)
    {
      print(paste("Number of bootstrap repetitions to redo: ",num.bad.clusters))
      beta.redo <- which(beta.matrix.final.reformat[,ncol(beta.matrix.final.reformat)-1]==1)
      beta.matrix.redo <- sapply(beta.redo,FUN=generate.importance.betas,raw.X=simulated.X,
                                 Y=Y,G=G,lambda.grid=lambda.grid,
                                 sample.weights=beta.importance,
                                 thresh=random.thresh,
                                 simplify=FALSE)
      
      beta.matrix.final.redo.reformat <- matrix(0,nrow=length(beta.redo),ncol=length(alpha_star)+length(beta_star)+2)
      for(k in 1:length(beta.redo)){beta.matrix.final.redo.reformat[k,] <- beta.matrix.redo[[k]]}
      
      beta.matrix.final.reformat[beta.redo,] <- beta.matrix.final.redo.reformat
      num.bad.clusters <- sum(beta.matrix.reformat[,ncol(beta.matrix.reformat)-1])
    }
    
    ### Save random CLUSSO lambda values
    random.CLUSSO.r2.lambda <- beta.matrix.final.reformat[,ncol(beta.matrix.final.reformat)]
    
    final.avg <- apply(beta.matrix.final.reformat,MARGIN=2,mean)
    
    ### Separate into final averaged alpha coefficient vector, final averaged beta coefficient
    ## vector, average bootstrap accuracy across 
    ## Don't want to use averaged alpha's, those should only be used for computing MSE
    ## within a given bootstrap iteration
    final.avg.beta <- final.avg[-c(1:length(alpha_star),
                                              (length(final.avg)-1):length(final.avg))]
    
    ##### NEW CODE 2/9/24 TO GET MSE BASED ON FINAL RANDOM CLUSSO ALPHA/BETA ESTIMATES
    ## Don't normalize beta_hat yet so you can compute the MSE, then normalize beta_hat 
    ## and compute the remaining performance metrics
    final.avg.alpha <- final.avg[1:length(alpha_star)]
    
    final.avg.beta.non.normalized <- final.avg.beta
    
    ## Threshold beta coefficients to zero that are smaller than 0.001 after L1 normalization
    final.avg.beta <- final.avg.beta/sum(abs(final.avg.beta))
    
    ### New code 3/5/24 to save non-normalized and normalized final beta coefficient 
    ### values averaged across bootstraps pre-thresholding
    ## First 10 
    beta.all.save <- c(final.avg.beta.non.normalized,final.avg.beta)
    
    small.inds <- abs(final.avg.beta)<random.thresh
    final.avg.beta[small.inds]=0
    
    final.avg.beta.non.normalized[small.inds]=0
    
    print(paste("Done with R2 bootstrapping for Random CLUSSO!"))
    
    ######### New code 4/14/24 to try to compute final MSE using what you call a structured OLS fit
    ### First, need to make a new set of design matrices that only include the selected by Random CLUSSO
    features.to.keep <- final.avg.beta.non.normalized!=0
    MSE.clust.X <- array(1, dim=c(G,sum(features.to.keep),n))
    
    for(i in 1:n)
    {
      MSE.clust.X[,,i] <- clust.X.CLUSSO[,features.to.keep==T,i]
    }
    
    ## Step 2, do structured OLS fit to get a new alpha hat and beta hat based on reduced set of features
    beta_init <- rnorm(dim(MSE.clust.X)[2])
    Random_CLUSSO_MSE_fit <- Mainfunction_albet_OLS(X_tr=MSE.clust.X,
                                                    Y_tr=Y,
                                                    alpha=alpha_init,
                                                    bet=beta_init)
    Random_CLUSSO_MSE_alpha <- Random_CLUSSO_MSE_fit[[1]]
    Random_CLUSSO_MSE_beta <- Random_CLUSSO_MSE_fit[[2]]
    
    exectime <- toc()
    time.for.Random.CLUSSO.after.clustering <- exectime$toc - exectime$tic
    
    ## beta_star doesn't matter because we're just returning MSE
    simulation.stuff <- simulation.results(Random_CLUSSO_MSE_beta,
                                           beta_star=beta_init,
                                           Random_CLUSSO_MSE_alpha,
                                           Y,
                                           MSE.clust.X)
    MSE.random <- simulation.stuff[4]
    
    ##################################################################################
    
    simulation.stuff <- simulation.results(final.avg.beta,beta_star,final.avg.alpha,Y,clust.X.CLUSSO)
    TPR.random <- simulation.stuff[1]
    FPR.random <- simulation.stuff[2]
    bias.random <- simulation.stuff[3]
    
    print(paste("Done with Random CLUSSO SOLS and perofrmance metrics!"))
    
    ###################################################################################
    performance.results <- rbind(c(TPR.clust,TPR.random,TPR.avg,TPR.full),
                                 c(FPR.clust,FPR.random,FPR.avg,FPR.full),
                                 c(bias.clust,bias.random,bias.avg,bias.full),
                                 c(MSE.clust,MSE.random,MSE.avg,MSE.full))  
      
    rownames(performance.results) <- c("TPR","FPR","L1 norm of bias","MSE")
    colnames(performance.results) <- c("CLUSSO","Random CLUSSO","Average balancing","Full information")
    
    return(list(performance.results,
                c(lambda.clust,lambda.full),
                cbind(random.CLUSSO.r1.lambda,random.CLUSSO.r2.lambda),beta.all.save))
  
}

# Compute simulation metrics 
simulation.results = function(beta_hat,beta_star,alpha_hat,Y,X)
{
  # L1 normalize beta_star
  beta_star <- beta_star/sum(abs(beta_star))
  
  TPR <- (length(intersect(which(beta_hat!=0),which(beta_star!=0)))/length(which(beta_star!=0)))*100
  FPR <- (length(intersect(which(beta_hat!=0),which(beta_star==0)))/length(which(beta_star==0)))*100
  bias <- sum(abs(beta_star-beta_hat))
  
  ########## For MSE, need to mean-center Y and the p x q predictors
  # Mean-centering but not scaling the variance for outcome
  Y <- scale(Y,scale=FALSE)
  
  # Get mean of the Xi's 
  X.mean <- apply(X, c(1,2), mean)
  
  SSE <- 0
  
  for(i in 1:(dim(X)[3]))
  {
    # Mean-center Xi's 
    X[,,i] <- X[,,i] - X.mean
    SSE <- SSE + (Y[i] - t(alpha_hat) %*% X[,,i] %*% beta_hat)^2
  }
  
  MSE <- (1/dim(X)[3]) * SSE
  #################################################################
  
  results <- c(TPR,FPR,bias,MSE)
  
  return(results)
}


# cor.structure is a string that indicates whether we want to use independent correlation structure, 
# exchangeable (rho = 0.6,0.9) correlation structure, or a block diagonal correlation structure 
generate.design.matrices = function(n,sigma_sq,alpha_star,beta_star,
                                    sigma.r,
                                    clust.X,clust.X.only.avg,w)
{
  
  simulated.X <- vector(mode = "list", length = n)
  X.avg <- matrix(0,nrow=n,ncol=length(beta_star))
  
  p <- sample(a:b,size=n,replace=TRUE)
  
  Y <- rep(0,n)
  
  # Record time for averaging across tubules for naive method
  tic()
  
  ## New code to compare true clusters you're drawing from to estimated cluster labels with GMM-based clustering
  latent.clust <- c()
  
  for(j in 1:n)
  {
    # Assuming that each subject has two truly informative tubules, 
    # One atrophic and one normal 
    # And rows of observed Xi are resampled from these two tubules with measurement
    # error
    Xi_star <- clust.X.only.avg[,,j]
  
    ### New code changes prob so that it samples normal/atrophic tubules in proportion 
    # To the true, set weights
    resamp.Xi.rows <- sample(1:nrow(Xi_star), size=p[j], replace=T,prob=c(w[j],1-w[j]))
    
    latent.clust <- c(latent.clust,resamp.Xi.rows)
    
    ###### resamp.Xi.rows contains true clustering/subpopulation assignments 
    resamp.Xi <- Xi_star[resamp.Xi.rows,]
    
    # Adding standard normal Gaussian measurement error to each resampled row
    resamp.Xi <- resamp.Xi + rnorm(nrow(resamp.Xi)*ncol(resamp.Xi),sd=sqrt(sigma.r))
    
    X <- resamp.Xi
    
    # Adding observed Xi to list of all Xi
    simulated.X[[j]] <- X
    
    Y[j] <- t(alpha_star) %*% clust.X[,,j] %*% beta_star + rnorm(n=1,mean=0,sd=sqrt(sigma_sq))
  
    X.avg[j,] <- colMeans(X)
    
  }
  
  exectime <- toc()
  time.to.avg <- exectime$toc - exectime$tic
  
  #################### New code to get feature matrices for CLUSSO on simulated feature matrices ###########
  # List of c x q matrices, where c is the number of clusters per subject
  # Minus 1 for the column count because you don't want to count the true 
  # cluster labels towards the final dimension of clust.X
  clust.X.CLUSSO <- array(1, dim=c(length(alpha_star),length(beta_star),n))
  
  ################### Need to cluster simulated observed feature matrices
  simulated.X.all <- do.call(rbind, simulated.X)
  
  # do clustering on all data
  NMM.out.simulated <- Mclust(simulated.X.all,G=length(alpha_star))$classification
  
  ###################### Picking the correspondence of latent cluster labels and estimated cluster labels that is most accurate 
  # Find best correspondence between two binary cluster label vectors

  # Method 1: Simple direct approach
  find_best_correspondence <- function(estimated, true) {
    # Calculate concordance for direct mapping (1->1, 2->2)
    direct_concordance <- sum(estimated == true) / length(estimated)
    
    # Calculate concordance for flipped mapping (1->2, 2->1)
    # Flip estimated labels: 1 becomes 2, 2 becomes 1
    estimated_flipped <- ifelse(estimated == 1, 2, 1)
    flipped_concordance <- sum(estimated_flipped == true) / length(true)
    
    # Return results
    if (direct_concordance >= flipped_concordance) {
      list(
        best_correspondence = "direct",
        concordance = direct_concordance,
        mapping = "1->1, 2->2",
        corrected_labels = estimated
      )
    } else {
      list(
        best_correspondence = "flipped", 
        concordance = flipped_concordance,
        mapping = "1->2, 2->1",
        corrected_labels = estimated_flipped
      )
    }
  }
  
  # Usage with your vectors - this will automatically adjust NMM.out.simulated
  result <- find_best_correspondence(NMM.out.simulated, latent.clust)
  
  # Print results
  cat("Best correspondence:", result$best_correspondence, "\n")
  cat("Mapping:", result$mapping, "\n") 
  cat("Concordance:", round(result$concordance, 3), "\n")
  
  # Automatically update NMM.out.simulated to have best correspondence
  NMM.out.simulated <- result$corrected_labels
  
  for(i in 1:n)
  {
    simulated.X[[i]] <- cbind(NMM.out.simulated[1:nrow(simulated.X[[i]])],simulated.X[[i]])
    NMM.out.simulated <- NMM.out.simulated[(nrow(simulated.X[[i]])+1):length(NMM.out.simulated)]
  }
  
  
  # Binary indicator as to whether simulation should be thrown out because 
  # At least one subject had no tubules in one of the clusters
  no.clust <- F
  
  for(i in 1:n)
  {
    ####################################################################### 
    ### Second column has estimated cluster labels, that correspond to how
    ### tubules were clustered using regular CLUSSO
    clust1.X.CLUSSO <- simulated.X[[i]][simulated.X[[i]][,1]==1,]
    clust2.X.CLUSSO <- simulated.X[[i]][simulated.X[[i]][,1]==2,]
    
    ## Need to separately consider the cases where only one object is 
    ## identified for one of the clusters - then it's a vector and not a matrix
    ## One tubule/vector is enough to do CLUSSO, so still want to proceed with CLUSSO
    # if either or both of the clust.X matrices are vectors
    if(is.vector(clust1.X.CLUSSO) & is.vector(clust2.X.CLUSSO) | 
       is.vector(clust1.X.CLUSSO) & is.matrix(clust2.X.CLUSSO) | 
       is.matrix(clust1.X.CLUSSO) & is.vector(clust2.X.CLUSSO))
    {
      ## Apply statements won't work on vectors, so need to handle the cases
      ## separately for each combination of clust1.X and/or clust2.X being vectors
      if(is.vector(clust1.X.CLUSSO) & is.matrix(clust2.X.CLUSSO))
      {
        ## We know that clust1.X only has one row
        w1 <- 1/nrow(simulated.X[[i]])
        
        ## Account for clust1.X being a vector rather than a matrix
        clust.X.CLUSSO[,,i] <- rbind(w1*clust1.X.CLUSSO[2:length(clust1.X.CLUSSO)],
                              (1-w1)*apply(clust2.X.CLUSSO[,2:ncol(clust2.X.CLUSSO)],MARGIN=2,mean))  
      }
      
      if(is.matrix(clust1.X.CLUSSO) & is.vector(clust2.X.CLUSSO))
      {
        w1 <- nrow(clust1.X.CLUSSO)/nrow(simulated.X[[i]])
        
        clust.X.CLUSSO[,,i] <- rbind(w1*apply(clust1.X.CLUSSO[,2:ncol(clust1.X.CLUSSO)],MARGIN=2,mean),
                              (1-w1)*clust2.X.CLUSSO[2:length(clust2.X.CLUSSO)])
      }
      
      if(is.vector(clust1.X.CLUSSO) & is.vector(clust2.X.CLUSSO))
      {
        ## We know that clust1.X only has one row
        w1 <- 1/nrow(simulated.X[[i]])
        
        ## Account for clust1.X being a vector rather than a matrix
        clust.X.CLUSSO[,,i] <- rbind(w1*clust1.X.CLUSSO[2:length(clust1.X.CLUSSO)],
                              (1-w1)*clust2.X.CLUSSO[2:length(clust2.X.CLUSSO)])  
      }
    }
    
    ## No objects were identified in one of the clusters
    else if(is.null(dim(clust1.X.CLUSSO)) | 
            is.null(dim(clust2.X.CLUSSO)) | 
            sum(is.nan(clust1.X.CLUSSO)) > 0 |  
            sum(is.nan(clust2.X.CLUSSO)) > 0 | 
            nrow(clust1.X.CLUSSO)==0 | 
            nrow(clust2.X.CLUSSO)==0) 
    {
      no.clust <- T
      print(paste("For simulated feature matrix, no tubules in a cluster"))  
    }
    
    ## Both clust1.X.CLUSSO and clust2.X.CLUSSO are matrices as desired! 
    else
    {
      w1 <- nrow(clust1.X.CLUSSO)/nrow(simulated.X[[i]])
      
      clust.X.CLUSSO[,,i] <- rbind(w1*apply(clust1.X.CLUSSO[,2:ncol(clust1.X.CLUSSO)],MARGIN=2,mean),
                                   (1-w1)*apply(clust2.X.CLUSSO[,2:ncol(clust2.X.CLUSSO)],MARGIN=2,mean))
    }
  
  }
  
  return(list(Y,
              X.avg,simulated.X,
              clust.X.CLUSSO,
              time.to.avg,
              no.clust))
}

### New function to run CLUSSO and also return clustering accuracy
### X that is being input has the first column as the true cluster label
## and estimated cluster label
run.CLUSSO = function(X,Y,G,lambda.grid,thresh)
{
  n <- length(Y)
  
  # List of c x q matrices, where c is the number of clusters per subject
  # Minus 1 for the column count because you don't want to count the true 
  # cluster labels towards the final dimension of clust.X
  clust.X.boot <- array(1, dim=c(G,ncol(X[[1]])-1,length(X)))
  
  # Binary indicator as to whether simulation should be thrown out because 
  # At least one subject had no tubules in one of the clusters
  no.clust <- F
  
  for(i in 1:n)
  {
    ####################################################################### 
    ### Second column has estimated cluster labels, that correspond to how
    ### tubules were clustered using regular CLUSSO
    clust1.X.boot <- X[[i]][X[[i]][,1]==1,]
    clust2.X.boot <- X[[i]][X[[i]][,1]==2,]
    
    ## Need to separately consider the cases where only one object is 
    ## identified for one of the clusters - then it's a vector and not a matrix
    ## One tubule/vector is enough to do CLUSSO, so still want to proceed with CLUSSO
    # if either or both of the clust.X matrices are vectors
    if(is.vector(clust1.X.boot) & is.vector(clust2.X.boot) | 
       is.vector(clust1.X.boot) & is.matrix(clust2.X.boot) | 
       is.matrix(clust1.X.boot) & is.vector(clust2.X.boot))
    {
      ## Apply statements won't work on vectors, so need to handle the cases
      ## separately for each combination of clust1.X and/or clust2.X being vectors
      if(is.vector(clust1.X.boot) & is.matrix(clust2.X.boot))
      {
        ## We know that clust1.X only has one row
        w1 <- 1/nrow(X[[i]])
        
        ## Account for clust1.X being a vector rather than a matrix
        clust.X.boot[,,i] <- rbind(w1*clust1.X.boot[2:length(clust1.X.boot)],
                              (1-w1)*apply(clust2.X.boot[,2:ncol(clust2.X.boot)],MARGIN=2,mean))  
      }
      
      if(is.matrix(clust1.X.boot) & is.vector(clust2.X.boot))
      {
        w1 <- nrow(clust1.X.boot)/nrow(X[[i]])
        
        clust.X.boot[,,i] <- rbind(w1*apply(clust1.X.boot[,2:ncol(clust1.X.boot)],MARGIN=2,mean),
                              (1-w1)*clust2.X.boot[2:length(clust2.X.boot)])
      }
      
      if(is.vector(clust1.X.boot) & is.vector(clust2.X.boot))
      {
        ## We know that clust1.X only has one row
        w1 <- 1/nrow(X[[i]])
        
        ## Account for clust1.X being a vector rather than a matrix
        clust.X.boot[,,i] <- rbind(w1*clust1.X.boot[2:length(clust1.X.boot)],
                              (1-w1)*clust2.X.boot[2:length(clust2.X.boot)])  
      }
    }
    
    ## No objects were identified in one of the clusters
    else if(is.null(dim(clust1.X.boot)) | 
            is.null(dim(clust2.X.boot)) | 
            sum(is.nan(clust1.X.boot)) > 0 |  
            sum(is.nan(clust2.X.boot)) > 0 | 
            nrow(clust1.X.boot)==0 | 
            nrow(clust2.X.boot)==0) 
    {
      no.clust <- T
      print(paste("For bootstrap iteration for random CLUSSO, no tubules in a cluster"))  
    }
    
    # For each subject, average tubules within each identified cluster and 
    # Make cluster ordering consistent across subjects
    
    # Check that there is at least one tubule that got classified into each 
    # cluster. Otherwise, going to drop subject
    
    # Each apply statement should produce a single row for the Xi generated by CLUSSO, and 
    # want to apply the appropriate weight w1 or 1-w1 for the respect normal/atrophic rows 
    
    else
    {
      ### New code to estimate weights for each row of final clust.X by computing 
      # The observed proportions of tubules belonging to cluster 1 and cluster 2 
      # from the EM/GMM clustering 
      w1 <- nrow(clust1.X.boot)/nrow(X[[i]])
      
      clust.X.boot[,,i] <- rbind(w1*apply(clust1.X.boot[,2:ncol(clust1.X.boot)],MARGIN=2,mean),
                            (1-w1)*apply(clust2.X.boot[,2:ncol(clust2.X.boot)],MARGIN=2,mean))
    }
  }
  
  ## After updated design matrices are generated, then run structured lasso to get
  ## Beta coefficients
  
  ### If one subject has no objects in one of the clusters, then can't do 
  ## CLUSSO fit
  if(!no.clust)
  {
    ########### CLUSSO RESULTS #################################################
    alpha_init <- rnorm(dim(clust.X.boot)[1])
    beta_init <- rnorm(dim(clust.X.boot)[2])
    
    ## Need to check on code to get CV-based lambda working
    ## You're thinking that you just might not be using enough subjects right now
    ############ CV-BASED LAMBDA ########################################################
    mse.clust.lambda <- sapply(lambda.grid,lambda.CV.mse,X.train=clust.X.boot,
                               Y.train=Y,
                               alpha_init=alpha_init,beta_init=beta_init,
                               thresh=random.thresh)
    lambda.clust <- lambda.grid[which(mse.clust.lambda==min(mse.clust.lambda))][1]
    ####################################################################################
    
    clust.fit <- Mainfunction_albet(clust.X.boot,Y,alpha_init,beta_init,lambda.clust,random.thresh)
    alpha_hat.clust <- clust.fit[[1]]  
    beta_hat.clust <- clust.fit[[2]]  
  } else{
    alpha_hat.clust <- rep(0,dim(clust.X.boot)[1])
    beta_hat.clust <- rep(0,dim(clust.X.boot)[2])
    lambda.clust <- 0
  }
  
  ## Storing no.clust as an integer so that it can be in the same matrix as 
  ## Beta coefficient values and clustering accuracy values
  return(list(alpha_hat.clust,beta_hat.clust,
              as.numeric(no.clust),clust.X.boot,lambda.clust))
  
}

### New function for Step 1 of Random CLUSSO: 1st round of bootstrapping
## b is the current bootstrap iteration, don't actually need it for the actual 
# function but just need an index to parallelize over
# Sample weights is a new argument that you will set to all zero if you're doing the 
## first round of bootstrapping and to the importance weights if you're doing the second
## round of bootstrapping
generate.importance.betas = function(b,raw.X,Y,G,lambda.grid,sample.weights,thresh)
{
  print(paste("On a new bootstrap!"))
  ## Draw a bootstrap sample of size n with replacement from original training
  # set
  bootstrap.ind <- sample(1:length(raw.X),size=length(raw.X),replace=T)  
  
  new.X <- raw.X[bootstrap.ind]
  new.Y <- Y[bootstrap.ind]
  
  ## Randomly choose q1 candidate covariates
  ## Note that first column is the true cluster labels, 
  # and the second column is the estimated cluster labels, so we need to 
  ## Pass these on for run CLUSSO and we are not selecting these columns/hence 
  ## -2 in formula for q1
  # boot.scale <- 1.5
  # q1 <- floor(boot.scale*sqrt(ncol(raw.X[[1]])-2))
  
  ### New code 2/10/24 to see what happens when we subsample all features ####
  ## q1 <- ncol(raw.X[[1]])-2
  
  #### New code 3/23/24 to see what happens when I sample floor(sqrt(q)) = 4 at a time
  ## assuming/currently using q=20 
  q1 <- floor(sqrt(ncol(raw.X[[1]])-1))
  
  
  if(sum(sample.weights)==0)
  {
    sub.covar <- sort(sample(2:ncol(raw.X[[1]]),size=q1,replace=F))  
  }
  
  else
  {
    # print(paste("Using interesting weights!"))
    sub.covar <- sort(sample(2:ncol(raw.X[[1]]),size=q1,replace=F,
                        prob=sample.weights/sum(abs(sample.weights))))
  }
  
  for(i in 1:length(new.X))
  {
    ## Make sure to carry over true and estimated cluster labels in addition to subsampled covariates
    new.X[[i]] <- new.X[[i]][,c(1,sub.covar)]
    
    ##### Second scheme for sampling tubules:
    ### Like for subjects, bootstrap p_i tubules with replacement
    bootstrap.obj <- sample(1:nrow(new.X[[i]]),
                            size=nrow(new.X[[i]]),replace=T)
    new.X[[i]] <- new.X[[i]][bootstrap.obj,]
    
  }
  
  ## Run CLUSSO
  bootstrap.CLUSSO.est <- run.CLUSSO(new.X,new.Y,G,lambda.grid,thresh)
  bootstrap.alpha <- bootstrap.CLUSSO.est[[1]]
  bootstrap.beta <- bootstrap.CLUSSO.est[[2]]
  no.clust <- bootstrap.CLUSSO.est[[3]]
  clust.X.boot <- bootstrap.CLUSSO.est[[4]]
  lambda.clust <- bootstrap.CLUSSO.est[[5]]
  
  ### Note that boostrap.beta was only run on some of the original covariates
  ## Any covariates not subsampled should be set to zero
  ## Have to subtract two because the first column is the true cluster labels
  ## and second 
  beta.final <- rep(0,ncol(raw.X[[1]])-1)
  
  # Was previously sampling from column three and beyond, 
  # but only have length of 
  # beta entries in your final beta coefficient vector, so need to map 
  # indices back to 1:length(beta) by subtracting 2 for the true and estimated
  # clustering labels
  sub.covar <- sub.covar - 1
  
  beta.final[sub.covar] <- bootstrap.beta
  
  ### Last element of vector is clustering accuracy, not beta
  ### Saving the results in this way just to allow for easier parallelization
  return(c(bootstrap.alpha,beta.final,no.clust,lambda.clust))
}

# Do structured lasso fit on training Xi's and Yi's and then compute MSE 
# using structured lasso fit on test Xi's and Yi's 
slasso.mse = function(X.train,Y.train,X.test,Y.test,alpha_init,beta_init,lambda,thresh)
{
  slasso_results <- Mainfunction_albet(X.train,Y.train,alpha_init,beta_init,lambda,thresh)
  beta_hat <- slasso_results[[2]]
  alpha_hat <- slasso_results[[1]]
  
  # Mean-centering test Y 
  Y.test <- scale(Y.test,scale=FALSE)
  
  # Get mean of the Xi's 
  X.mean.test <- apply(X.test, c(1,2), mean)
  
  # Compute MSE
  SSE <- 0
  
  for(i in 1:(dim(X.test)[3]))
  {
    # Mean-center Xi's 
    X.test[,,i] <- X.test[,,i] - X.mean.test
    SSE <- SSE + (Y.test[i] - t(alpha_hat) %*% X.test[,,i] %*% beta_hat)^2
  }
  
  MSE <- (1/dim(X.test)[3]) * SSE
  
  return(MSE)
}

# Make folds for cross-validation for picking lambda for structured lasso
# Takes in number of subjects which are being used for the training dataset
CV.make.folds = function(nTrain)
{
  # Compute sizes for each fold
  sampleSizeF <- floor(1/5 * nTrain)
  
  # Picking indices for each fold 
  indicesF1 <- sort(sample(1:nTrain,size=sampleSizeF))
  indicesNotF <- setdiff(1:nTrain,indicesF1)
  
  indicesF2 <- sort(sample(indicesNotF, size=sampleSizeF))
  indicesNotF <- setdiff(1:nTrain,c(indicesF1,indicesF2))
  
  indicesF3 <- sort(sample(indicesNotF, size=sampleSizeF))
  indicesNotF <- setdiff(1:nTrain,c(indicesF1,indicesF2,indicesF3))
  
  indicesF4 <- sort(sample(indicesNotF, size=sampleSizeF))
  indicesF5 <- setdiff(indicesNotF,indicesF4)
  
  return(list(indicesF1,indicesF2,indicesF3,indicesF4,indicesF5))
}

# Get cross-validated MSE using structured lasso for a particular lambda value
lambda.CV.mse = function(X.train,Y.train,alpha_init,beta_init,lambda,thresh){
  
  # Define the different folds for each round of cross-validation
  nfolds <- 5
  folds <- CV.make.folds(dim(X.train)[3])
  CV.fold.matrix <- diag(nfolds)
  mse.vec <- rep(0,nfolds)
  
  for(i in 1:length(folds))
  {
    fold.setup <- CV.fold.matrix[i,]
    
    # Make training/testing split
    # Four folds are for training and one fold is for testing
    train.fold <- which(fold.setup==0)
    test.fold <- which(fold.setup==1)
    cv.train.ind <- sort(c(unlist(folds[train.fold])))
    cv.test.ind <- sort(c(unlist(folds[test.fold])))
    
    # Compute MSE for given training/testing split
    mse.vec[i] <- slasso.mse(X.train[,,cv.train.ind],Y.train[cv.train.ind],X.train[,,cv.test.ind],
                             Y.train[cv.test.ind],alpha_init,beta_init,lambda,thresh)
  }
  
  return(mean(mse.vec))
  
}