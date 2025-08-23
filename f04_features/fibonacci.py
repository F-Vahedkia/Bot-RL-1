    


# ---- Fibonacci retracement & Extension -----------------------------------
    @staticmethod
    def fibonacci_retracement(high: pd.Series, low: pd.Series, lookback: int = 100, 
                              levels: list = None) -> pd.DataFrame:
        """
        محاسبه سطوح فیبوناچی (Retracement) بر اساس آخرین نوسان (Swing High و Swing Low).
        
        Parameters
        ----------
        high : pd.Series
            سری داده‌های High (بالاترین قیمت هر کندل).
        low : pd.Series
            سری داده‌های Low (پایین‌ترین قیمت هر کندل).
        lookback : int, default=100
            تعداد کندل‌های گذشته که برای یافتن Swing High و Swing Low بررسی می‌شود.
        levels : list, optional
            لیست سطوح فیبوناچی دلخواه. به طور پیش‌فرض [0.236, 0.382, 0.5, 0.618, 0.786].

        Returns
        -------
        pd.DataFrame
            شامل سطوح فیبوناچی برای آخرین بازه انتخاب‌شده.
            ستون‌ها: ['Level', 'Price']
        """
        
        # اگر سطوح دلخواه داده نشد، از سطوح استاندارد استفاده می‌کنیم
        if levels is None:
            levels = [0.236, 0.382, 0.5, 0.618, 0.786]

        # بررسی داده کافی
        if len(high) < lookback or len(low) < lookback:
            raise ValueError("داده‌های کافی برای محاسبه فیبوناچی وجود ندارد.")

        # انتخاب آخرین n کندل
        recent_high = high[-lookback:].max()
        recent_low = low[-lookback:].min()

        # تشخیص جهت حرکت (صعودی یا نزولی)
        uptrend = recent_high > recent_low

        # محاسبه سطوح فیبوناچی
        fibonacci_levels = []
        for level in levels:
            if uptrend:
                price_level = recent_high - (recent_high - recent_low) * level
            else:
                price_level = recent_low + (recent_high - recent_low) * level
            fibonacci_levels.append((level, price_level))

        # تبدیل خروجی به DataFrame مرتب
        fib_df = pd.DataFrame(fibonacci_levels, columns=["Level", "Price"])
        return fib_df

        # مثال استفاده:
        # fib = Indicators.fibonacci_retracement(df['High'], df['Low'], lookback=120)
        # print(fib)

    # ---- Fibonacci Levels ----------------------------------------------------
    @staticmethod
    def fibonacci_levels(high: pd.Series, low: pd.Series, lookback: int = 100,
                         levels: list = None, include_extension: bool = True) -> pd.DataFrame:
        """
        محاسبه سطوح Fibonacci Retracement و (اختیاری) Extension.
        
        Parameters
        ----------
        high : pd.Series
            سری داده‌های High (بالاترین قیمت هر کندل)
        low : pd.Series
            سری داده‌های Low (پایین‌ترین قیمت هر کندل)
        lookback : int
            تعداد کندل‌های گذشته برای تعیین Swing High/Low
        levels : list, optional
            لیست سطوح Fibonacci Retracement، پیش‌فرض [0.236,0.382,0.5,0.618,0.786]
        include_extension : bool
            اگر True باشد، سطوح Extension نیز محاسبه می‌شوند (1.272,1.618,2.0)

        Returns
        -------
        pd.DataFrame
            DataFrame شامل سطوح Fibonacci، نوع (Retracement/Extension) و قیمت
        """
        # سطوح پیش‌فرض Retracement
        if levels is None:
            levels = [0.236, 0.382, 0.5, 0.618, 0.786]

        # بررسی داده کافی
        if len(high) < lookback or len(low) < lookback:
            raise ValueError("داده‌های کافی برای محاسبه فیبوناچی وجود ندارد.")

        # انتخاب آخرین n کندل
        recent_high = high[-lookback:].max()
        recent_low = low[-lookback:].min()

        # تعیین جهت روند
        uptrend = recent_high > recent_low

        fib_list = []

        # محاسبه سطوح Retracement
        for level in levels:
            price = recent_high - (recent_high - recent_low) * level if uptrend else recent_low + (recent_high - recent_low) * level
            fib_list.append({"Type": "Retracement", "Level": level, "Price": price})

        # محاسبه سطوح Extension (اختیاری)
        if include_extension:
            ext_levels = [1.272, 1.618, 2.0]
            for ext in ext_levels:
                price = recent_high + (recent_high - recent_low) * (ext - 1) if uptrend else recent_low - (recent_high - recent_low) * (ext - 1)
                fib_list.append({"Type": "Extension", "Level": ext, "Price": price})

        fib_df = pd.DataFrame(fib_list)
        fib_df.sort_values(by="Price", inplace=True, ascending=not uptrend)
        fib_df.reset_index(drop=True, inplace=True)

        return fib_df

        # مثال استفاده:
        # df = pd.DataFrame({'High': high_series, 'Low': low_series})
        # fibs = Indicators.fibonacci_levels(df['High'], df['Low'], lookback=120)
        # print(fibs)


class MovingAverageCross_old:
    """
    A flexible Moving Average Crossover detector.
    Supports both SMA and EMA with arbitrary period pairs.
    Generates binary signals: 
        +1 for Bullish crossover (Golden Cross),
        -1 for Bearish crossover (Death Cross),
        0 for no signal.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50, ma_type: str = "EMA"):
        """
        Initialize the crossover detector.
        
        Args:
            fast_period (int): Period for the fast moving average.
            slow_period (int): Period for the slow moving average.
            ma_type (str): Type of MA to use ("SMA" or "EMA").
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be smaller than slow period.")
        if ma_type not in ["SMA", "EMA"]:
            raise ValueError("ma_type must be either 'SMA' or 'EMA'.")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.upper()

    def _calc_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Helper: calculate SMA or EMA."""
        if self.ma_type == "SMA":
            return series.rolling(window=period, min_periods=period).mean()
        elif self.ma_type == "EMA":
            return series.ewm(span=period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Generate moving average crossover signals.

        Args:
            df (pd.DataFrame): Input OHLC dataframe with a 'close' column (or given price_col).
            price_col (str): Column name for price series.

        Returns:
            pd.DataFrame: Original dataframe with added columns:
                - fast_MA
                - slow_MA
                - MA_Cross_Signal  (+1, -1, or 0)
        """
        df = df.copy()

        # Calculate fast and slow MAs
        df["fast_MA"] = self._calc_ma(df[price_col], self.fast_period)
        df["slow_MA"] = self._calc_ma(df[price_col], self.slow_period)

        # Detect crossovers
        df["MA_Cross_Signal"] = 0
        df.loc[(df["fast_MA"] > df["slow_MA"]) & (df["fast_MA"].shift(1) <= df["slow_MA"].shift(1)), "MA_Cross_Signal"] = +1
        df.loc[(df["fast_MA"] < df["slow_MA"]) & (df["fast_MA"].shift(1) >= df["slow_MA"].shift(1)), "MA_Cross_Signal"] = -1

        return df


class MovingAverageCross:
    """
    Robust & extensible MA crossover detector.

    Usage:
        mac = MovingAverageCross(fast=12, slow=26, ma_type='ema',
                                 signal_col='ma_cross_signal',
                                 min_bars_between_signals=3,
                                 confirm_bars=1)
        df = mac.add_to_df(df)   # returns df with fast/slow MA cols and signal column
        signal = mac.last_signal(df)  # +1 buy, -1 sell, 0 none

    Parameters:
    - fast, slow: int or List[int] (if lists, multiple pairs will be computed)
    - ma_type: 'ema' or 'sma'
    - signal_col: name of final signal column
    - min_bars_between_signals: suppress repeat signals within this many bars
    - confirm_bars: require signal direction to hold for `confirm_bars` (>=1) bars to confirm
    - magnitude: if True returns magnitude of crossover (diff normalized) instead of only -1/0/+1
    """

    def __init__(
        self,
        fast: Union[int, List[int]] = 12,
        slow: Union[int, List[int]] = 26,
        ma_type: str = "ema",
        signal_col: str = "ma_cross_signal",
        min_bars_between_signals: int = 3,
        confirm_bars: int = 1,
        magnitude: bool = False,
        price_col: str = "close",
    ):
        self.fast = [fast] if isinstance(fast, int) else list(fast)
        self.slow = [slow] if isinstance(slow, int) else list(slow)
        if len(self.fast) != len(self.slow):
            # allow broadcasting: if one of lists length 1, broadcast it
            if len(self.fast) == 1:
                self.fast = self.fast * len(self.slow)
            elif len(self.slow) == 1:
                self.slow = self.slow * len(self.fast)
            else:
                raise ValueError("fast and slow must have same length or one must be scalar.")
        self.ma_type = ma_type.lower()
        if self.ma_type not in ("ema", "sma"):
            raise ValueError("ma_type must be 'ema' or 'sma'")
        self.signal_col = signal_col
        self.min_bars_between_signals = max(0, int(min_bars_between_signals))
        self.confirm_bars = max(1, int(confirm_bars))
        self.magnitude = bool(magnitude)
        self.price_col = price_col
