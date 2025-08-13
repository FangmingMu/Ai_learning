# R7: 生产级部署与监控 🛡️

## 🎯 模块核心目标

本模块的核心目标是**掌握将一个高性能AI应用，安全、稳定、可维护地部署到生产环境的全流程**。我们将学习和实践现代化的DevOps与MLOps理念，为我们的RAG系统建立起自动化的部署流水线、完善的监控告警体系和基本的安全防护机制。完成本模块后，你将拥有独立负责一个AI应用从开发到上线再到运维的完整闭环能力。

## 🧠 核心概念

1.  **CI/CD (持续集成/持续部署):** 将“写好代码”到“上线服务”之间的所有步骤（测试、构建、部署）完全自动化。这是现代软件工程的基石，旨在实现快速、可靠的迭代。
2.  **容器编排 (Container Orchestration):** `docker-compose`适合本地开发，但在生产环境中，我们需要更强大的工具（如Kubernetes）来管理应用的扩缩容、自愈和负载均衡。
3.  **可观测性 (Observability):** 这不仅仅是“监控”。它包含了三个支柱：**Metrics（指标）**、**Logs（日志）**和**Traces（链路追踪）**。我们的目标是让系统不再是一个“黑盒”，它的任何风吹草动我们都能了如指掌。
4.  **LLM应用安全 (LLM Application Security):** 除了传统的网络安全，我们还需要关注针对LLM的特殊攻击，如Prompt注入、数据泄露等。

## 🛠️ 技术栈与工具

*   **容器化:** (复用) `Docker`, `docker-compose`
*   **CI/CD:** `GitHub Actions`
*   **监控与告警:** `Prometheus`, `Grafana`, `Alertmanager`
*   **Web服务器:** `Gunicorn` / `Uvicorn`

---

## 🚀 行动计划：打造永不宕机的AI服务

### **步骤一：生产级容器化 (Production-Grade Containerization)**

为部署做最后的打包准备。

*   [ ] **7.1. 学习并使用Gunicorn/Uvicorn:**
    *   **学习思想:** FastAPI自带的开发服务器不适合在生产环境使用。我们需要一个专业的ASGI服务器，如`Uvicorn`（由`Gunicorn`管理），来运行我们的FastAPI应用，以获得更好的性能和稳定性。
    *   **实践任务:** 修改你的`Dockerfile`，将启动命令从`python main.py`改为使用`gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`。

*   [ ] **7.2. 编写生产环境的`docker-compose.prod.yml`:**
    *   **实践任务:** 创建一个区别于开发环境的生产部署文件。它应该包含你在`R6`中拆分出的所有微服务（Web Gateway, RAG Core, vLLM等），并配置好网络和服务依赖关系。

### **步骤二：实现CI/CD自动化流水线 (Automation Pipeline)**

让代码自己“走”到服务器上。

*   [ ] **7.3. 学习GitHub Actions基础:**
    *   **学习思想:** 了解`.github/workflows/`目录下的`YAML`文件是如何定义自动化工作流的。学习其核心概念：`event`（触发事件）, `job`（任务）, `step`（步骤）, `action`（动作）。

*   [ ] **7.4. 编写CI流水线 (Continuous Integration):**
    *   **实践任务:** 创建一个`ci.yml`工作流。该工作流由`push`到`main`分支触发。
    *   **核心步骤:**
        1.  **代码检出 (Checkout Code)**
        2.  **设置Python环境 (Setup Python)**
        3.  **安装依赖 (Install Dependencies)**
        4.  **运行代码风格检查与单元测试 (Lint & Test)** - (如果你在之前实现了测试)

*   [ ] **7.5. 编写CD流水线 (Continuous Deployment):**
    *   **实践任务:** 创建一个`cd.yml`工作流。该工作流可以在`CI`成功后自动触发，或者通过手动触发（`workflow_dispatch`）。
    *   **核心步骤:**
        1.  **登录镜像仓库 (Login to Docker Hub)**
        2.  **构建并推送Docker镜像 (Build and Push Docker Image)** - 将你的应用打包成带版本号的镜像，并推送到Docker Hub。
        3.  **(选修) 远程部署:** 通过SSH连接到你的云服务器，执行`docker-compose pull`和`docker-compose up -d`来拉取最新镜像并重启服务。

### **步骤三：建立监控告警体系 (Monitoring & Alerting)**

为你的AI应用装上“心电监护仪”。

*   [ ] **7.6. 学习并部署Prometheus + Grafana:**
    *   **学习思想:** Prometheus是一个“拉”模型的监控系统，它会定期从你的服务暴露的接口上抓取指标数据。Grafana则负责将这些数据以酷炫的图表展示出来。
    *   **实践任务:** 在你的`docker-compose.prod.yml`中，增加`prometheus`和`grafana`两个服务。并为FastAPI应用添加一个`prometheus-fastapi-instrumentator`中间件，使其能自动暴露` /metrics`接口。

*   [ ] **7.7. 创建核心业务监控面板:**
    *   **实践任务:** 登录到Grafana界面，创建一个新的Dashboard。至少添加以下几个核心监控图表：
        1.  **API请求数 (QPS)**
        2.  **API请求延迟 (P99/P95 Latency)**
        3.  **API错误率 (Error Rate)**
        4.  **LLM调用平均耗时**

*   [ ] **7.8. (选修)配置告警规则:**
    *   学习在Prometheus中编写告警规则，并配置Alertmanager，实现当“API错误率连续5分钟超过5%”时，能收到邮件或钉钉通知。

### **步骤四：安全与反馈 (Security & Feedback)**

让应用更健壮，并形成闭环。

*   [ ] **7.9. 实践基本的Prompt安全防护:**
    *   **学习思想:** 了解什么是Prompt注入攻击。学习一些基本的防御策略，例如在Prompt模板中加入明确的边界指示符，或者对用户输入进行过滤。
    *   **实践任务:** 设计一个恶意的Prompt（如“忽略你之前的指令，现在你是一个...”），测试你的RAG系统是否会“上当”。然后，尝试通过优化Prompt来修复这个漏洞。

*   [ ] **7.10. 建立用户反馈循环:**
    *   **实践任务:** 在你的Streamlit前端界面上，为每一个AI生成的答案，添加一个“👍”和“👎”的按钮。
    *   当用户点击时，将对应的“问题-上下文-答案-用户反馈”完整地记录到一个数据库或日志文件中。这为我们持续优化模型和评估系统，提供了最宝贵的数据来源，将流程拉回了`R1`，形成了一个完整的、持续进化的闭环。

---

## ✅ 最终产出

当你完成这个终极模块后，你的`R7/`文件夹下应该包含：

1.  `docker-compose.prod.yml`: 你的生产级服务编排文件。
2.  `prometheus/prometheus.yml`: 你的监控配置文件。
3.  `grafana/provisioning/`: 你的Grafana仪表盘配置文件。
4.  `.github/workflows/`: 你的CI/CD自动化工作流。
5.  `FINAL_DEPLOYMENT_REPORT.md`: 一份最终的报告文档，包含：
    *   你的生产环境系统架构图。
    *   展示你的Grafana监控仪表盘的截图。
    *   分享你对CI/CD流程的理解，以及它如何提升开发效率。
    *   你对LLM应用安全的思考和实践。

## 🎉 祝贺！

你已经走完了从0到1，再到企业级部署的全过程。你现在拥有的，不仅仅是几个酷炫的项目，而是一整套**系统性的、工业级的AI应用开发与运维能力**。这份GitHub仓库，就是你通往任何一家顶级公司的、最有力的“敲门金砖”。