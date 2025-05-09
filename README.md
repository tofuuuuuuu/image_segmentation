# Curvilinear Feature Detection Project

Implements a graph-based image segmentation algorithm using Minimum Spanning Trees (MST). The primary goal is to detect edge pixels in an image (typically corresponding to boundaries between distinct objects) and represent each detected edge as a graph edge using only two data points: the starting and ending pixel positions.

The algorithm enables efficient representation and processing of image data. 

Inspired by a lecture on Youtube. 

Hair removal algorithm is a modification of this paper: 
[Curvilinear feature extraction using minimum spanning trees](https://www.sciencedirect.com/science/article/abs/pii/0734189X84902214)

## Sample 1
<figure>
  <img
  src="https://github.com/user-attachments/assets/becad51f-8d05-458a-a1ce-bfc4aa23ef32"
  alt="Figure 1.">
  <figcaption> 
  image dimensions: 182 x 273
  Pixel count: 49686
  Final edge count : 3386
  </figcaption>
</figure>

## Sample 2
<figure>
  <img
  src="https://github.com/user-attachments/assets/b8810e9c-5a11-4243-b2fa-118d8c8bce73"
  alt="Figure 2.">
  <figcaption> 
  Image dimensions: 184 x 270
  Pixel count: 49680
  Final edge count : 3187
  </figcaption>
</figure>

## Sample 3
<figure>
  <img
  src="https://github.com/user-attachments/assets/1c0f34de-27ba-4794-83cc-5b6046947a52"
  alt="Figure 3.">
  <figcaption> 
  Image dimensions: 184 x 271
  Pixel count: 49864
  Final edge count : 3221
  </figcaption>
</figure>

## Sample 4
<figure>
  <img
  src="https://github.com/user-attachments/assets/8ed926f0-dcce-4dbb-9305-acb1df1ec335"
  alt="Figure 4.">
  <figcaption> 
  Image dimensions: 182 x 273
  Pixel count: 49686
  Final edge count : 3086
  </figcaption>
</figure>
