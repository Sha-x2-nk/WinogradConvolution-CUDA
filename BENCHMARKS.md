## Comparison Against cuDNN
The following table compares the performance of this implementation against cuDNN:
|GPU|Platform|Build System|
|--|--|--|
|Nvidia RTX 3070Ti Laptop GPU|Windows 11|Visual Studio 22|

We strongly feel cuDNN also uses 4x4_3x3 implementation of winograd.

<!-- </br> -->
<table>
  <tr>
    <th>[N, C, H, W, K, P]</th>
    <th>Algorithm</th>
    <th>Filter transform</th>
    <th>Input transform</th>
    <th>Hadamard Product</th>
    <th>Inverse Transform</th>
    <th>Total(w/o filter transform)</th>
    <th>Max diff in val</th>
  </tr>
  <tr>
    <td rowspan="3">[1, 512, 28, 28, 512, 1]</td>
    <td>2x2_3x3</td>
    <td>95.04 us</td>
    <td>43.49 us</td>
    <td>350.43 us</td>
    <td>41.89 us</td>
    <td><b>435.81 us</b></td>
    <td>0</td>
    
  </tr>
  <tr>
    <td>4x4_3x3</td>
    <td>183.81 us</td>
    <td>26.43 us</td>
    <td>460.54 us</td>
    <td>24.74 us</td>
    <td>511.71 us</td>
    <td>0.03</td>
  </tr>
  <tr>
    <td>cuDNN Non Fused</td>
    <td>130 us</td>
    <td>169.07 us</td>
    <td>2.47 ms</td>
    <td>162.88 us</td>
    <td>2.8 ms</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td rowspan="3">[1, 256, 56, 56, 256, 1]</td>
    <td>2x2_3x3</td>
    <td>28.32 us</td>
    <td>78.56 us</td>
    <td>299.52 us</td>
    <td>72.61 us</td>
    <td>450.69 us</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4x4_3x3</td>
    <td>51.55 us</td>
    <td>48.96 us</td>
    <td>242.27 us</td>
    <td>40.58 us</td>
    <td><b>331.81 us</b></td>
    <td>0.08</td>
  </tr>
  <tr>
    <td>cuDNN Non Fused</td>
    <td>30.94 us</td>
    <td>334.78 us</td>
    <td>2.3 ms</td>
    <td>315.36 us</td>
    <td>2.95 ms</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td rowspan="3">[1, 128, 112, 112, 128, 1]</td>
    <td>2x2_3x3</td>
    <td>10.34 us</td>
    <td>143.65 us</td>
    <td>315.81 us</td>
    <td>132.51 us</td>
    <td>591.97 us</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4x4_3x3</td>
    <td>18.14 us</td>
    <td>89.89 us</td>
    <td>194.30 us</td>
    <td>66.34 us</td>
    <td><b>350.53 us</b></td>
    <td>0.03</td>
  </tr>
  <tr>
    <td>cuDNN Non Fused</td>
    <td>9.06 us</td>
    <td>676.96 us</td>
    <td>2.36 ms</td>
    <td>620.86 us</td>
    <td>3.65 ms</td>
    <td>0.01</td>
  </tr>

  <tr>
    <td rowspan="3">[20, 512, 28, 28, 512, 1]</td>
    <td>2x2_3x3</td>
    <td>95.04 us</td>
    <td>766.11 us</td>
    <td>5.73 ms</td>
    <td>700.54 us</td>
    <td>7.19 ms</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4x4_3x3</td>
    <td>181.34 us</td>
    <td>416.86 us</td>
    <td>3.6 ms</td>
    <td>344.03 us</td>
    <td><b>4.36 ms</b></td>
    <td>0.2</td>
  </tr>
  <tr>
    <td>cuDNN Non Fused</td>
    <td>130 us</td>
    <td>426.34 us</td>
    <td>4.6 ms</td>
    <td>387.23 us</td>
    <td>5.41 ms</td>
    <td>0.03</td>
  </tr>
  <tr>
    <td rowspan="3">[20, 256, 56, 56, 256, 1]</td>
    <td>2x2_3x3</td>
    <td>28.32 us</td>
    <td>1.46 ms</td>
    <td>5.3 ms</td>
    <td>1.3 ms</td>
    <td>8.06 ms</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4x4_3x3</td>
    <td>51.2 us</td>
    <td>945.34 us</td>
    <td>3.29 ms</td>
    <td>669.86 us</td>
    <td><b>4.9 ms</b></td>
    <td>0.09</td>
  </tr>
  <tr>
    <td>cuDNN Non Fused</td>
    <td>30.94 us</td>
    <td>844.16 us</td>
    <td>4.42 ms</td>
    <td>773.09 us</td>
    <td>6.03 ms</td>
    <td>0.02</td>

  </tr>
  <tr>
    <td rowspan="3">[20, 128, 112, 112, 128, 1]</td>
    <td>2x2_3x3</td>
    <td>10.34 us</td>
    <td>2.76 ms</td>
    <td>5.48 ms</td>
    <td>2.49 ms</td>
    <td>10.73 ms</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4x4_3x3</td>
    <td>18.14 us</td>
    <td>1.68 ms</td>
    <td>3.21 ms</td>
    <td>1.18 ms</td>
    <td><b>6.07 ms</b></td>
    <td>0.04</td>

  </tr>
  <tr>
    <td>cuDNN Non Fused</td>
    <td>9.06 us</td>
    <td>1.69 ms</td>
    <td>4.71 ms</td>
    <td>1.55 ms</td>
    <td>7.95 ms</td>
    <td>0.01</td>
  </tr>

</table>

These are only kernel runtimes, benched by Nvidia Nsight Compute, total runtimes can vary than sum of them. For ex, since filter and input transforms are independent, we can compute them together and synchronize when we do hadamard.