فایل‌های پیشنهادی پروژه (حداقل‌های پایه) ربات تریدر Bot-RL-1
(هر مورد = مسیر/فایل — توضیح کوتاه)
1.	requirements.txt / Pipfile — لیست پکیج‌ها و نسخه‌ها (pin شده).
2.	.gitignore —   جلوگیری از commit شدن credentials و فایل‌های بزرگ.
3.	README.md —   توضیح پروژه، نحوه اجرا، ساختار پوشه.
4.	f01_config/config.yaml —   فایل پیکربندی مرکزی (credentials، symbols، paths، rl params).
5.	f10_utils/config_loader.py —   لودر ایمن و قابل reload کانفیگ.
6.	f10_utils/logging_cfg.py —   پیکربندی مرکزی logging (console + file + rotating).
7.	f02_data/mt5_connector.py — (  اتصال به MT5، get_candles، retry، shutdown).
8.	f02_data/mt5_data_loader.py — (batch download, ذخیره CSV، مدیریت خطا).
9.	f02_data/fetch_data.py —   اسکریپت اجرایی برای کرون/زمان‌بندی (entrypoint برای دانلود).
10.	f02_data/data_handler.py —   لایهٔ آماده‌سازی داده برای آموزش/تحلیل (range fetch, DataFrame).
11.	f03_env/trading_env.py —   محیط Gym/Gymnasium برای RL (reset/step/render).
12.	f05_agents/agent.py —   پیاده‌سازی یا wrapper برای الگوریتم RL (PPO/SAC).
13.	f07_training/train.py —   لوپ آموزش، checkpoint، رزوم/ادامه.
14.	f07_training/utils.py —   ابزارهای مربوط به آموزش (schedules, lr, augment).
15.	f06_replay_buffer/ —   پیاده‌سازی replay buffer (در صورت نیاز prioritized).
16.	f08_evaluation/backtest.py —   بک‌تست استراتژی‌ها و محاسبه معیارها.
17.	f08_evaluation/metrics.py —   محاسبه Sharpe, CAGR, Drawdown, Win/Loss.
18.	f09_execution/executor_3.py —   ارسال سفارشات زنده، بررسی تایید اجرا، مدیریت SL/TP.
19.	risk/risk_manager.py —   محاسبه حجم بر اساس ریسک، سقف ضرر روزانه.
20.	f04_features/indicators.py —   محاسبه EMA/RSI/ATR/MACD و استخراج featureها.
21.	models/ —   پوشه نگهداری مدل‌های ذخیره‌شده (checkpoint files).
22.	f12_notebooks/ —   نوت‌بوک‌های آنالیز داده و مصورسازی.
23.	f11_tests/ —   تست‌های واحد برای هر ماژول (CI قابل اجرا).
24.	ci/.github/workflows/ci.yml —   تنظیمات GitHub Actions (lint, tests).
25.	docs/ —   مستندات معماری، API و راهنمای توسعه.
26.	monitoring/alerts.py —   اعلان خطا/هشدار (تلگرام/ایمیل).
27.	scripts/run_scheduler.sh   یا systemd service — برای اجرای زمان‌بندی‌شده.
28.	examples/ —   اسکریپت‌های نمونه (مثلاً run_fetch, run_train, run_live).

اسامی فایل هایی که تا الان نوشتیم:
1.	config.yaml        ,    config_example.yaml
2.	config_loader.py
.env                                         شخصی است و شامل یوزر و پسوورد ورود به اکانت حساب در آلپاری می باشد
3.	mt5_connector.py
4.	mt5_data_loader.py
5.	data_handler.py
6.	fetch_data.py
7.	logging_cfg.py
8.	risk_manager.py
9.	executor.py
10.	trading_env.py
