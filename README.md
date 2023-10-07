# Denture biometrics

This repository contains all of the code used for the project on automating the analysis of dental radiographies in order to take biometrical measurements of the dental pieces, by J.C. Perez-Ramirez, under the supervision of Dr. F.C Cuevas de la Rosa.

# Description

biometria.py segments the intermaxilar region via a thresholding filter that highlights the regions of lesser bone density. With this information, the quadratic curve that best fits the intermaxilar region is obtained via a regresion algorithm, and the image intensity along a family of such curves detects the aproximate roots of the denture.
