import pandas as pd
from typing import List, Dict, Tuple


class Fibonacci:
    """
    Utility class for generating Fibonacci retracement, extension,
    and projection levels.

    Notes:
        - All outputs are returned as Pandas DataFrames for consistency.
        - Designed to be easily expandable with new Fibonacci-based tools.
    """

    # Common Fibonacci ratios
    RETRACEMENT_RATIOS: List[float] = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSION_RATIOS: List[float] = [1.272, 1.618, 2.0, 2.618]
    PROJECTION_RATIOS: List[float] = [0.618, 1.0, 1.618]

    @staticmethod
    def calculate_retracement(high: float, low: float) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels between a swing high and low.

        Args:
            high (float): Swing high price.
            low (float): Swing low price.

        Returns:
            pd.DataFrame: retracement levels with ratio and price.
        """
        diff = high - low
        levels = [(ratio, high - ratio * diff) for ratio in Fibonacci.RETRACEMENT_RATIOS]

        return pd.DataFrame(
            levels, columns=["ratio", "price"]
        ).assign(method="retracement", description="Swing High-Low")

    @staticmethod
    def calculate_extension(high: float, low: float) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels (beyond retracement).

        Args:
            high (float): Swing high price.
            low (float): Swing low price.

        Returns:
            pd.DataFrame: extension levels with ratio and price.
        """
        diff = high - low
        levels = [(ratio, high + ratio * diff) for ratio in Fibonacci.EXTENSION_RATIOS]

        return pd.DataFrame(
            levels, columns=["ratio", "price"]
        ).assign(method="extension", description="Swing High-Low")

    @staticmethod
    def calculate_projection(A: float, B: float, C: float) -> pd.DataFrame:
        """
        Calculate Fibonacci projection levels (ABC pattern).

        Args:
            A (float): Point A price.
            B (float): Point B price.
            C (float): Point C price.

        Returns:
            pd.DataFrame: projection levels with ratio and price.
        """
        diff = B - A
        levels = [(ratio, C + ratio * diff) for ratio in Fibonacci.PROJECTION_RATIOS]

        return pd.DataFrame(
            levels, columns=["ratio", "price"]
        ).assign(method="projection", description="ABC Pattern")

    @staticmethod
    def multi_timeframe_confluence(data: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Calculate retracement levels for multiple timeframes
        and return combined results.

        Args:
            data (Dict[str, Tuple[float, float]]):
                Dictionary with timeframe as key and (high, low) as value.

        Example:
            {
                "1h": (1800, 1750),
                "4h": (1850, 1700)
            }

        Returns:
            pd.DataFrame: all levels with timeframe column included.
        """
        all_levels = []
        for timeframe, (high, low) in data.items():
            df = Fibonacci.calculate_retracement(high, low)
            df = df.assign(timeframe=timeframe)
            all_levels.append(df)

        return pd.concat(all_levels, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    retr = Fibonacci.calculate_retracement(2000, 1900)
    print("\nRetracement Levels:\n", retr)

    ext = Fibonacci.calculate_extension(2000, 1900)
    print("\nExtension Levels:\n", ext)

    proj = Fibonacci.calculate_projection(1900, 2000, 1950)
    print("\nProjection Levels:\n", proj)

    mtf = Fibonacci.multi_timeframe_confluence({"1h": (2000, 1900), "4h": (2050, 1850)})
    print("\nMulti-Timeframe Retracement:\n", mtf)
