from typing import Any, Dict, Optional, Union

import pandas as pd
from dateutil.relativedelta import relativedelta
from typeguard import typechecked

from utils.math import normalize
from utils.portfolio.asset import Asset

from .null_tax_model import NullTaxModel


class MABollingerPortfolio:
    @typechecked()
    def __init__(
        self,
        setup: Dict[str, Dict[str, Union[str, float]]],
        bollinger_k: float = 2.0,
        start_value: float = 10000,
        rebalancing: Optional[relativedelta] = None,
        rebalancing_offset: Optional[relativedelta] = None,
        detailed_output: bool = False,
        details_memory: Optional[Dict[str, Any]] = None,
        spread=0,
        tax_model=NullTaxModel(),
    ):
        assert len(setup.keys()) >= 1, "You must specify at least one ETF."
        assert len(setup.keys()) == len(set(setup.keys())), (
            "Every ETF must be unique in your portfilio."
        )
        for name, v in setup.items():
            assert "dist" in v, "Every asset needs a key 'dist'!"
            if "ma" not in v or v["ma"] == 1:
                v["ma"] = 1
                v["ma_asset"] = name
            assert "ma_asset" in v, "Every asset needs a key 'ma_asset'!"
        assert sum([v["dist"] for v in setup.values()]) <= 100, (
            f"Your Portfolio has an allocation of {sum([d for d in setup.values()])}%"
        )

        self._setup = setup
        self._bollinger_k = bollinger_k
        self._start_value = start_value
        self._detailed_output = detailed_output
        self._details_memory = details_memory if details_memory is not None else {}
        self._rebalancing = rebalancing
        self._rebalancing_offset = rebalancing_offset
        self._spread = spread / 2
        self._tax_model = tax_model

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a MA+Bollinger backtest:
        1. When price falls below its moving average, set a 'wait for BB' flag.
        2. If, while below the MA, price falls BELOW the lower Bollinger, SELL.
        3. If price rises back above the MA before selling, reset the flag.
        4. Buying: When price is above MA (original logic).
        Returns:
            pd.DataFrame: Daily portfolio value breakdown by asset and total sum.
        """

        asset_names = list(self._setup.keys())
        assets = {}
        self._values = {}
        self._details_memory["asset"] = {
            name: dict(buys=[], sells=[]) for name in asset_names
        }

        # Precompute moving averages and Bollinger bands
        mas = {
            name: data[self._setup[name]["ma_asset"]]
            .rolling(window=self._setup[name]["ma"])
            .mean()
            for name in asset_names
        }

        boll_bands = {}
        for name in asset_names:
            ma_series = mas[name]
            asset_series = data[self._setup[name]["ma_asset"]]
            std_series = asset_series.rolling(window=self._setup[name]["ma"]).std()
            boll_k = self._bollinger_k
            upper_band = ma_series + boll_k * std_series
            lower_band = ma_series - boll_k * std_series
            boll_bands[name] = dict(
                upper=upper_band,
                lower=lower_band,
            )

        max_ma_length = max(v["ma"] for v in self._setup.values())

        for name in asset_names:
            assert name in data.columns, (
                f"Asset with the name {name} does not exist in data ({data.columns})."
            )
            self._values[name] = (self._start_value * self._setup[name]["dist"]) / 100
            assets[name] = Asset(name, detailed_output=False)

        portfolio_values = pd.DataFrame(
            index=data.index[max_ma_length:], columns=asset_names + ["sum"]
        )

        # Prepare MA wait state variables
        ma_below_flag = {name: False for name in asset_names}

        # Prepare next rebalancing event, if used
        if self._rebalancing is not None:
            next_rebalancing = (
                portfolio_values.index[0] + self._rebalancing + self._rebalancing_offset
            )

        for i in portfolio_values.index:
            # a) Rebalancing if due
            if self._rebalancing is not None and i > next_rebalancing:
                next_rebalancing = next_rebalancing + self._rebalancing
                self._do_rebalancing(assets, data.loc[i, :])

            # b) Per-asset main logic
            for name in asset_names:
                setup = self._setup[name]
                asset_price = data.loc[i, name]
                ma_value = mas[name].loc[i]
                compare_price = data.loc[i, setup["ma_asset"]]
                lower_band = boll_bands[name]["lower"].loc[i]

                # ---- 1) Reset wait flag and BUY if price above MA
                if compare_price >= ma_value:
                    # BUY logic if not currently invested
                    if self._values[name] is not None:
                        # print(f"buy: {i}")
                        real_price = asset_price * (1 + self._spread)
                        amount = self._values[name] / real_price
                        self._log(
                            f"[{i}] Buy {amount:.2f}x {name} at ${real_price:.2f} (Uptrend, MA={ma_value:.2f})"
                        )
                        assets[name].buy(amount, real_price)
                        self._details_memory["asset"][name]["buys"].append(i)
                        self._values[name] = None
                    # Always reset wait flag if price above MA
                    ma_below_flag[name] = False

                # ---- 2) Set wait flag if crosses below MA (only once)
                elif compare_price < ma_value and ma_below_flag[name] is False:
                    ma_below_flag[name] = True  # price crossed below MA

                # ---- 3) SELL if in wait-state, invested, and falls below lower BB band
                if (
                    ma_below_flag[name]
                    and self._values[name] is None  # invested
                    and compare_price < lower_band
                ):
                    # print(f"sell: {i}")
                    real_price = asset_price * (1 - self._spread)
                    amount = assets[name].amount
                    proceeds = amount * real_price
                    self._log(
                        f"[{i}] Sell {amount:.2f}x {name} at ${real_price:.2f} (Below MA, lower BB breach)"
                    )
                    _, gain = assets[name].sell(amount, real_price)
                    self._details_memory["asset"][name]["sells"].append(i)
                    self._tax_model.add_gain(name, gain)
                    self._values[name] = proceeds
                    ma_below_flag[name] = False  # reset after selling

                # ---- 4) Record value
                value_held = assets[name].amount * asset_price
                cash = self._get_value(name)
                portfolio_values.loc[i, name] = value_held + cash

            # c) Taxes if any
            while self._tax_model.open_tax > 1.0:
                self._sell(assets, data.loc[i, :], self._tax_model.open_tax)

        # ---- Finalization ----
        portfolio_values["sum"] = portfolio_values[asset_names].sum(axis=1)

        # Store chart data for visualization
        self._details_memory["chart"] = {}
        for name in asset_names:
            self._details_memory["chart"][f"{name}_ma"] = mas[name]
            self._details_memory["chart"][f"{name}_ma_asset"] = data[
                self._setup[name]["ma_asset"]
            ]
            self._details_memory["chart"][name] = data[name]
            self._details_memory["chart"][f"{name}_value"] = normalize(
                portfolio_values[name], data[name]
            )
            # Include Bollinger bands for visualization
            self._details_memory["chart"][f"{name}_boll_upper"] = boll_bands[name][
                "upper"
            ]
            self._details_memory["chart"][f"{name}_boll_lower"] = boll_bands[name][
                "lower"
            ]

        return portfolio_values

    def _sell(self, assets, prices, target: float):
        # (unchanged)
        self._log(f" * Sell assets to get ${target:.2f} for tax.")
        sum_value = sum(
            [
                (prices[name] * asset.amount + self._get_value(name))
                for name, asset in assets.items()
            ]
        )
        for name, asset in assets.items():
            value = asset.amount * prices[name] + self._get_value(name)
            percent = (value / sum_value) * 100
            asset_target = (target * percent) / 100
            if asset_target < 0.1:
                self._log(
                    f"Ignore selling [{name}] since amount ${asset_target:.2f} is too small."
                )
                continue

            if self._values[name] is None:
                amount = asset_target / prices[name]
                self._log(
                    f"Sell {amount} (from {assets[name].amount}) of [{name}] to pay ${asset_target:.2f} of tax."
                )
                _, gain = asset.sell(amount, prices[name])
                self._tax_model.pay_tax(name, asset_target)
                self._tax_model.add_gain(name, gain)

            else:
                assert self._values[name] >= asset_target
                self._values[name] -= asset_target
                self._tax_model.pay_tax(name, asset_target)

    def _do_rebalancing(self, assets: Dict[str, Asset], prices: pd.Series):
        # (unchanged)
        self._log(f"** Rebalancing: {prices.name}")

        sum_value = sum(
            [
                (prices[name] * asset.amount) + self._get_value(name)
                for name, asset in assets.items()
            ]
        )
        for name, asset in assets.items():
            value = asset.amount * prices[name] + self._get_value(name)
            percent = (value / sum_value) * 100
            diff = percent - self._setup[name]["dist"]
            self._log(
                f" * current state [{name}]: ${value:.2f} (percent: {percent:.2f}%, diff: {diff:.2f}%)"
            )

            target_value = (self._setup[name]["dist"] * sum_value) / 100
            diff_value = value - target_value

            if diff_value > 0:
                if self._values[name] is None:
                    asset_price = prices[name] * (1 - self._spread)
                    amount = diff_value / asset_price
                    self._log(
                        f" => Sell {amount:.2f}x {name} for ${asset_price:.2f} each (total: ${amount * asset_price:.2f})"
                    )
                    _, gain = asset.sell(amount, asset_price)
                    self._tax_model.add_gain(name, gain)
                else:
                    self._log(f" => Reallocate ${diff_value:.2f} from {name} away")
                    self._values[name] -= diff_value

            elif diff_value < 0:
                if self._values[name] is None:
                    asset_price = prices[name] * (1 + self._spread)
                    amount = -diff_value / asset_price
                    self._log(
                        f" => Buy {amount:.2f}x {name} for ${asset_price:.2f} each (total: ${amount * asset_price:.2f})"
                    )
                    asset.buy(amount, asset_price)

                else:
                    self._log(f" => Reallocate ${-diff_value:.2f} to {name}")
                    self._values[name] -= diff_value

            value = asset.amount * prices[name] + self._get_value(name)
            percent = (value / sum_value) * 100
            diff = percent - self._setup[name]["dist"]
            if self._detailed_output:
                self._log(
                    f"  ==> {name}: ${value:.2f} (percent: {percent:.2f}%, diff: {diff:.2f}%)"
                )

    def _log(self, msg):
        if self._detailed_output:
            print(msg)

    def _get_value(self, name):
        return self._values[name] if self._values[name] is not None else 0
