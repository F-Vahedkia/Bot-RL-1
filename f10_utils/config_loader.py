# f10_utils/config_loader.py
"""
ConfigLoader:
- بارگذاری config از f01_config/config.yaml
- اعمال override از متغیرهای محیطی (env vars) در اولویت
- API ساده: get(key, default=None), get_all(copy=False), reload()
"""
from typing import Any, Dict, Iterable
import os
import yaml
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_path: str = None, env_prefix: str = "BOT_"):
        """
        config_path: مسیر فایل yaml (اگر None مسیر پیش‌فرض استفاده می‌شود)
        env_prefix: پیشوندی که برای نام‌گذاری env vars استفاده می‌شود (مثلاً BOT_)
        """
        # مسیر پیش‌فرض نسبی به ریشه پروژه (دو سطح بالاتر از این فایل)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        if config_path is None:
            config_path = os.path.join(base_dir, "f01_config", "config.yaml")

        # --- load .env from project root if it exists ---
        env_path = os.path.join(base_dir, ".env")
        if os.path.exists(env_path):
            try:
                load_dotenv(dotenv_path=env_path)
            except Exception:
                logger.exception("Failed to load .env from %s", env_path)
        # -------------------------------------------------

        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config = {}
        self.reload()

    # ---------- بارگذاری اولیه ----------
    def _load_yaml(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

        if not isinstance(cfg, dict):
            raise ValueError("Loaded configuration is not a valid dictionary.")
        return cfg

    # ---------- کمکی: تولید نام env برای مسیر کلیدها ----------
    def _env_name_for_path(self, path: Iterable[str]) -> str:
        # path مثال: ["mt5_credentials", "login"] -> BOT_MT5_CREDENTIALS_LOGIN
        parts = [p.upper() for p in path]
        return f"{self.env_prefix}{'_'.join(parts)}"

    # ---------- کمکی: تبدیل رشته env به نوع مناسب ----------
    def _cast_env_value(self, val_str: str, original_value: Any):
        s = val_str.strip()
        # bool
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # int
        try:
            iv = int(s)
            return iv
        except Exception:
            pass
        # float
        try:
            fv = float(s)
            return fv
        except Exception:
            pass
        # list (comma separated) — اگر original یک لیست بوده یا در env کاما داشته باشد
        if "," in s:
            return [item.strip() for item in s.split(",") if item.strip() != ""]
        # fallback: رشته
        return s

    # ---------- کمکی: بازگشتن دیکشنری و جایگزینی با env ----------
    def _apply_env_overrides(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        برای هر مسیر در cfg، اگر env var معادل تعریف شده بود، مقدار را override می‌کند.
        این تابع ساختار دیکشنری را حفظ می‌کند و فقط مقادیر برگ را جایگزین می‌کند.
        """
        def recurse(node, path):
            if isinstance(node, dict):
                res = {}
                for k, v in node.items():
                    res[k] = recurse(v, path + [k])
                return res
            else:
                # مسیر کامل => نام env
                env_name = self._env_name_for_path(path)
                env_val = os.getenv(env_name)
                if env_val is not None:
                    # تبدیل براساس محتوا
                    try:
                        casted = self._cast_env_value(env_val, node)
                        logger.debug("Overriding config %s with env %s = %r", ".".join(path), env_name, casted)
                        return casted
                    except Exception:
                        logger.exception("Failed to cast env var %s", env_name)
                        return node
                # همچنین چک کوتاه برای نام بدون پیشوند (برای راحتی)
                env_name_no_prefix = "_".join([p.upper() for p in path])
                env_val2 = os.getenv(env_name_no_prefix)
                if env_val2 is not None:
                    try:
                        return self._cast_env_value(env_val2, node)
                    except Exception:
                        return node
                return node

        return recurse(cfg, [])

    # ---------- API عمومی ----------
    def reload(self):
        """یکبار YAML را دوباره می‌خواند و اوور رایتهای env را اعمال می‌کند."""
        raw = self._load_yaml()
        self.config = self._apply_env_overrides(raw)
        logger.info("Config loaded from %s (env prefix=%s)", self.config_path, self.env_prefix)

    def get(self, key: str, default=None):
        """دسترسی به top-level keys: get('mt5_credentials')"""
        return self.config.get(key, default)

    def get_all(self, copy: bool = False):
        """بازگردانی کل config؛ اگر copy=True یک کپی سطحی برمی‌گرداند."""
        return dict(self.config) if copy else self.config
        
# یک instance آماده برای import راحت
config = ConfigLoader().get_all()

def load_config(path: str = None):
    """
    Compatibility helper:
    اگر path داده شود از آن استفاده می‌کند، در غیر این صورت از مسیر پیش‌فرض استفاده می‌کند.
    برمی‌گرداند: dict کانفیگ
    """
    return ConfigLoader(config_path=path).get_all()
