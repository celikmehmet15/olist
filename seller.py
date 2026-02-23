import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data["sellers"].copy()
        sellers.drop("seller_zip_code_prefix", axis=1, inplace=True)
        sellers.drop_duplicates(inplace=True)
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        order_items = self.data["order_items"].copy()
        orders = self.data["orders"].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on="order_id")

        ship.loc[:, "shipping_limit_date"] = pd.to_datetime(ship["shipping_limit_date"])
        ship.loc[:, "order_delivered_carrier_date"] = pd.to_datetime(ship["order_delivered_carrier_date"])
        ship.loc[:, "order_delivered_customer_date"] = pd.to_datetime(ship["order_delivered_customer_date"])
        ship.loc[:, "order_purchase_timestamp"] = pd.to_datetime(ship["order_purchase_timestamp"])

        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date) / np.timedelta64(24, "h")
            )
            return days if days > 0 else 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp) / np.timedelta64(24, "h")
            )
            return days

        delay = ship.groupby("seller_id").apply(delay_to_logistic_partner).reset_index()
        delay.columns = ["seller_id", "delay_to_carrier"]

        wait = ship.groupby("seller_id").apply(order_wait_time).reset_index()
        wait.columns = ["seller_id", "wait_time"]

        return delay.merge(wait, on="seller_id")

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        orders_approved = self.data["orders"][["order_id", "order_approved_at"]].dropna()

        orders_sellers = (
            orders_approved.merge(self.data["order_items"], on="order_id")[
                ["order_id", "seller_id", "order_approved_at"]
            ]
            .drop_duplicates()
        )
        orders_sellers["order_approved_at"] = pd.to_datetime(orders_sellers["order_approved_at"])

        orders_sellers["date_first_sale"] = orders_sellers["order_approved_at"]
        orders_sellers["date_last_sale"] = orders_sellers["order_approved_at"]

        df = orders_sellers.groupby("seller_id").agg(
            {"date_first_sale": "min", "date_last_sale": "max"}
        )
        df["months_on_olist"] = round(
            (df["date_last_sale"] - df["date_first_sale"]) / np.timedelta64(30, "D")
        )
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data["order_items"]

        n_orders = order_items.groupby("seller_id")["order_id"].nunique().reset_index()
        n_orders.columns = ["seller_id", "n_orders"]

        quantity = order_items.groupby("seller_id", as_index=False).agg({"order_id": "count"})
        quantity.columns = ["seller_id", "quantity"]

        result = n_orders.merge(quantity, on="seller_id")
        result["quantity_per_order"] = result["quantity"] / result["n_orders"]
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return (
            self.data["order_items"][["seller_id", "price"]]
            .groupby("seller_id")
            .sum()
            .rename(columns={"price": "sales"})
        )

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """
        reviews = self.data["order_reviews"][["order_id", "review_score"]].dropna()
        order_seller = self.data["order_items"][["order_id", "seller_id"]].drop_duplicates()

        df = order_seller.merge(reviews, on="order_id", how="inner")
        if df.empty:
            return None

        out = (
            df.groupby("seller_id")["review_score"]
            .agg(
                review_score="mean",
                share_of_one_stars=lambda s: (s == 1).mean(),
                share_of_five_stars=lambda s: (s == 5).mean(),
            )
            .reset_index()
        )
        return out

    def get_training_data(self):
        """
        Legacy training set (kept for backward compatibility).
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist',
        'share_of_one_stars', 'share_of_five_stars', 'review_score',
        'n_orders', 'quantity', 'quantity_per_order', 'sales']
        """
        training_set = (
            self.get_seller_features()
            .merge(self.get_seller_delay_wait_time(), on="seller_id")
            .merge(self.get_active_dates(), on="seller_id")
            .merge(self.get_quantity(), on="seller_id")
            .merge(self.get_sales(), on="seller_id")
        )

        review_df = self.get_review_score()
        if review_df is not None:
            training_set = training_set.merge(review_df, on="seller_id", how="left")

        return training_set

    # =========================
    # CEO PROJECT (NEW METHODS)
    # =========================

    def get_review_score_with_costs(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score', 'cost_of_reviews'
        """
        reviews = self.data["order_reviews"][["order_id", "review_score"]].dropna()
        order_seller = self.data["order_items"][["order_id", "seller_id"]].drop_duplicates()

        df = order_seller.merge(reviews, on="order_id", how="inner")
        if df.empty:
            return None

        df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

        cost_map = {1: 100, 2: 50, 3: 40}
        df["review_cost"] = df["review_score"].map(cost_map).fillna(0)

        out = (
            df.groupby("seller_id")["review_score"]
            .agg(
                review_score="mean",
                share_of_one_stars=lambda s: (s == 1).mean(),
                share_of_five_stars=lambda s: (s == 5).mean(),
            )
            .reset_index()
        )

        costs = (
            df.groupby("seller_id")["review_cost"]
            .sum()
            .reset_index()
            .rename(columns={"review_cost": "cost_of_reviews"})
        )

        out = out.merge(costs, on="seller_id", how="left")
        out["cost_of_reviews"] = out["cost_of_reviews"].fillna(0)
        return out

    def get_training_data_ceo(self):
        """
        CEO training set:
        adds 'cost_of_reviews', 'revenues', 'profits'
        """
        ts = self.get_training_data().copy()

        review_costs = self.get_review_score_with_costs()
        if review_costs is not None:
            ts = ts.merge(
                review_costs[["seller_id", "cost_of_reviews"]],
                on="seller_id",
                how="left",
            )
        else:
            ts["cost_of_reviews"] = 0

        ts["cost_of_reviews"] = ts["cost_of_reviews"].fillna(0)

        # revenues = subscription fees + sales fees
        months = ts["months_on_olist"].fillna(0)
        billed_months = np.maximum(1, np.ceil(months))
        ts["revenues"] = 0.10 * ts["sales"] + 80 * billed_months

        ts["profits"] = ts["revenues"] - ts["cost_of_reviews"]
        return ts
