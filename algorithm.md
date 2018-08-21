---
description: Explain every method use in this tool
---

# Technique

## Projection Method

### TSNE

This method is appropriate for clustering.We can see clearly that similar data points gather together to from groups. However, this good result is in the price of computation efficiency.

In this project , TSNE is being used in data scatter plot.

{% hint style="info" %}
This computation efficiency could be solved by using cuda version of t-sne implementation.  
[https://github.com/CannyLab/tsne-cuda](https://github.com/CannyLab/tsne-cuda)
{% endhint %}

### PCA

Fast computation time is an advantage of PCA method. By using eigenvalue decomposition,  
even though large datasets could be directly compute PCA transformation.  
  
In this project, PCA in used in image query to prevent high dimension distance calculation.

 

## Image Query

Below image show the pipeline of image retrieval, four major method can be seen in this image.We use CNN/Pre-trained single-pass method.

1. First, we forward all the images in our datasets to get feature vector\( the second-last dense layer\).
2. Captured feature vector could be high dimension vector,so PCA is needed to scale down dimension into 100  or fewer.
3. Given a query image , first get it feature vector with PCA down sample,and then calculate distance matrix.
4. Sort distance matrix , and proposed neatest k result.

![SIFT Meets CNN: A Decade Survey of Instance Retrieval](https://lh4.googleusercontent.com/3pI3zzFsCtwzt4tLzGehl8M9pUKFHrp4dX7C6VndJafIU7584VnV2HacTY-YuJ5IkgvhUgoAb1tbnVLp_ageAvw-sz1-N-hoY5FgNlrqv2oNUjyAVAPP6qezz95-bg2XLyeWlfwE8F8)

