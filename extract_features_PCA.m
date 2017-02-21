function [A_f, B_f] = extract_features_PCA(A,B,dim)

    %%
    %% PCA - get principal components
    %%
    [eigen_vecs, eigen_values, Mean_Image] = PCA(A, dim);
                 
    %% subtract mean image from the faces and project onto selected
    %% number of principal components
    %%
  
    A_f = eigen_vecs(:,1:dim)'* (A-repmat(Mean_Image,1,size(A,2)));   
    B_f = eigen_vecs(:,1:dim)'* (B-repmat(Mean_Image,1,size(B,2)));    
