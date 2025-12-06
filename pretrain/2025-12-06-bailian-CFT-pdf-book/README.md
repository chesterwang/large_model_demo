
## 准备数据

运行 `prepare_data.py` 解析pdf教材文件，输出纯文本jsonl文件，格式如下。

![alt text](image.png)


## continue pre-training

1. 通过阿里云百炼平台，上传数据集，启动CPT训练任务。
2. 运行时间：

![alt text](image2.png)

**运行结果如下**

这里eval_loss和eval_ppl的换算我想除了正则项目之外还有指数底数不同，ppl使用2底数。

```Python
2025-12-06 20:26:57,402 - INFO - data download succeeded, start to pre-processor
2025-12-06 20:28:19,002 - INFO - data pre-process succeeded, start to fine-tune
Training start!
Estimated time: 7.333333333333334 mins
Estimated token: 393216
{'learning_rate': 0.0, 'consumed_train_samples': 48, 'consumed_train_tokens': 393216, 'global_step/max_steps': '3/3', 'loss': 1.00644}
{'eval_loss': 1.1319, 'eval_ppl': 3.1014094824472096, 'global_step/max_steps': '3/3'}
Actual number of consumed tokens is 393216!
Training completed
2025-12-06 20:36:58,049 - INFO - fine-tuned output got, start to transfer it for inference
2025-12-06 20:39:07,094 - INFO - transfer for inference succeeded, start to deliver it for inference
2025-12-06 20:41:37,187 - INFO - start to save checkpoint
2025-12-06 20:44:39,609 - INFO - finetune-job succeeded
2025-12-06 20:44:40,108 - INFO - ##FT_COMPLETE##
2025-12-06 20:44:40,093 - INFO - training usage 393216
```
