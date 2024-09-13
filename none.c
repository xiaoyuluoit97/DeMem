#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

// 创建 MAX30102
MAX30105 particleSensor;

// 心率和 SpO2 计算相关变量
const byte RATE_SIZE = 4; // 存储心率值的数组大小
byte rates[RATE_SIZE];     // 存储最新的心率值
byte rateSpot = 0;         // 存储心率数组的位置
long lastBeat = 0;         // 上次检测到心跳的时间

float beatsPerMinute;
int beatAvg;

// SpO2 相关变量
float SpO2;
float validity;

void setup() {
  Serial.begin(115200);
  while (!Serial); // 串口初始化

  Serial.println("Initializing...");

  // 初始化传感器
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1);
  }

  Serial.println("Place your finger on the sensor to begin...");

  // 配置传感器
  particleSensor.setup(); // 使用默认设置
  particleSensor.setPulseAmplitudeRed(0x0A); // 设置红色LED强度
  particleSensor.setPulseAmplitudeGreen(0);   // 关闭绿色LED
}

void loop() {
  long irValue = particleSensor.getIR(); // 读取红外值
  long redValue = particleSensor.getRed(); // 读取红色值

  if (checkForBeat(irValue) == true) {
    // 检测到心跳，计算 BPM
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20) {
      rates[rateSpot++] = (byte)beatsPerMinute; // 存储心率值
      rateSpot %= RATE_SIZE;

      // 计算平均 BPM
      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) {
        beatAvg += rates[x];
      }
      beatAvg /= RATE_SIZE;
    }
  }

  // 简单的 SpO2 计算
  if (redValue > 10000 && irValue > 50000) {
    SpO2 = 100.0 - ((float)redValue / (float)irValue) * 100.0;
    validity = 100.0;
  } else {
    SpO2 = 0.0;
    validity = 0.0;
  }

  // 显示结果
  Serial.print("IR=");
  Serial.print(irValue);
  Serial.print(", Red=");
  Serial.print(redValue);
  Serial.print(", BPM=");
  Serial.print(beatsPerMinute);
  Serial.print(", Avg BPM=");
  Serial.print(beatAvg);
  Serial.print(", SpO2=");
  Serial.print(SpO2);
  Serial.print(", Validity=");
  Serial.print(validity);

  if (irValue < 50000) {
    Serial.print(" No finger?");
  }

  Serial.println();
}
