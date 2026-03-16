# Python 人脸识别模块接口文档

## 1. 提取人脸特征 (Extract Feature)

- **URL**: `/extract_feature`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **描述**: 接收 Base64 编码的图片，检测人脸并返回 128 维特征向量。
- **请求参数**:
  ```json
  {
    "image": "base64_encoded_image_string"
  }
  ```
- **响应**:
  ```json
  {
    "code": 200,
    "msg": "success",
    "data": {
      "feature": "0.123, -0.456, ..." // 逗号分隔的 128 个浮点数
    }
  }
  ```
- **错误响应**:
  - 400: No image provided / Invalid image / No face detected
  - 500: Internal server error

## 2. 交互说明

本模块启动后会：
1. 自动从 `http://dlib.net/files/` 下载所需模型文件（如果不存在）。
2. 启动 Flask HTTP 服务（默认端口 5000）。
3. 打开本地摄像头窗口，进行实时人脸检测与识别。
4. 定期（60秒）调用 Java 后端接口 `/api/employee/features` 同步员工人脸特征库。
5. 识别成功后，调用 Java 后端接口 `/api/attendance/record` 和 `/api/capture/save`。

## 3. 目录结构

- `app.py`: 主程序
- `models/`: 存放 dlib 模型文件
- `venv/`: Python 虚拟环境
- `../data/`: 存放抓拍图片和录像 (相对于 python 目录的上级目录)
