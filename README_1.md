# Deep-JSCC-Image-Transmission

本项目是“现代编码与理论”课程的仿真实验，主要围绕深度联合信源信道编码（Deep JSCC）与传统分离信源信道编码（SSCC）的性能对比分析。

---

## 实现参考

- **Deep JSCC 网络实现**  
  本项目中的 Deep JSCC 网络架构参考了以下论文与代码：  
  *OFDM-guided deep joint source channel coding for wireless multipath fading channels*,  
  IEEE Transactions on Cognitive Communications and Networking, vol.8, 2022.  
  🔗 GitHub: [https://github.com/mingyuyng/OFDM-guided-JSCC](https://github.com/mingyuyng/OFDM-guided-JSCC)

- **SSCC 系统实现**  
  采用 IEEE 802.11n WiFi 标准的 LDPC 编码方案，参考以下开源项目：  
  🔗 GitHub: [https://github.com/tavildar/LDPC](https://github.com/tavildar/LDPC)

- **注意力机制引入方法**  
  引入 Attention 模块的 Deep JSCC 架构参考以下论文与代码：  
  *Wireless image transmission using deep source channel coding with attention modules*,  
  IEEE Transactions on Circuits and Systems for Video Technology, vol.32, 2021.  
  🔗 GitHub: [https://github.com/alexxu1988/ADJSCC](https://github.com/alexxu1988/ADJSCC)

---

## 仿真实验设计

我们设计了以下三组实验，对比 Deep JSCC 与传统 SSCC 方案的性能差异：

1. **相同 SPP（Symbol per Pixel）下，不同 SNR 条件下**  
   Deep JSCC 与 SSCC 的 PSNR/SSIM 性能对比；
2. **相同 SNR 条件下，不同 SPP 设置下**  
   Deep JSCC 与 SSCC 的 PSNR/SSIM 性能对比；
3. **相同 SPP 条件下**  
   Deep JSCC 与 Attention-based Deep JSCC 在不同 SNR 下的自适应鲁棒性对比；

---
