import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()
    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        orders = self.data["orders"].copy()

        cols = [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        df = orders[cols].copy()

        if is_delivered:
            df = df[df["order_status"] == "delivered"].copy()

        # datetime'a çevir
        for col in [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # ondalıklı gün cinsinden (NaN kalabilir)
        df["wait_time"] = (
            (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])
            .dt.total_seconds() / 86400
        )
        df["expected_wait_time"] = (
            (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"])
            .dt.total_seconds() / 86400
        )

        delay = df["wait_time"] - df["expected_wait_time"]

        # NaN'leri koru; negatifse 0, pozitifse delay
        df["delay_vs_expected"] = np.where(
            delay.isna(),
            np.nan,
            np.where(delay > 0, delay, 0.0)
        )

        return df[["order_id", "wait_time", "expected_wait_time", "delay_vs_expected", "order_status"]]



    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data["order_reviews"].copy()

        df = reviews[["order_id", "review_score"]].copy()

        df["dim_is_five_star"] = (df["review_score"] == 5).astype(int)
        df["dim_is_one_star"] = (df["review_score"] == 1).astype(int)

        return df[["order_id", "dim_is_five_star", "dim_is_one_star", "review_score"]]


    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        items = self.data["order_items"][["order_id"]].copy()

        df = (
            items.groupby("order_id")
            .size()
            .reset_index(name="number_of_items")
        )

        return df


    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        items = self.data["order_items"][["order_id", "seller_id"]].copy()

        df = (
            items.groupby("order_id")["seller_id"]
                 .nunique()
                 .reset_index(name="number_of_sellers")
        )

        return df


    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        items = self.data["order_items"][["order_id", "price", "freight_value"]].copy()

        df = (
            items.groupby("order_id", as_index=False)[["price", "freight_value"]]
                 .sum()
        )

        return df


    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        (average seller-customer distance per order, in km)
        """
        orders = self.data["orders"][["order_id", "customer_id"]].copy()
        customers = self.data["customers"][["customer_id", "customer_zip_code_prefix"]].copy()
        sellers = self.data["sellers"][["seller_id", "seller_zip_code_prefix"]].copy()
        items = self.data["order_items"][["order_id", "seller_id"]].copy()
        geo = self.data["geolocation"].copy()

        # 1) Zip prefix -> (lat,lng) için geolocation'ı zip bazında ortalama ile özetle
        geo_zip = (
            geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
               .mean()
               .reset_index()
        )

        # 2) Order -> customer zip
        order_customer = (
            orders.merge(customers, on="customer_id", how="left")
                  .rename(columns={"customer_zip_code_prefix": "customer_zip"})
        )

        # 3) Order -> seller zip (order_items üzerinden)
        order_seller = (
            items.merge(sellers, on="seller_id", how="left")
                 .rename(columns={"seller_zip_code_prefix": "seller_zip"})
        )

        # 4) Order-seller satırına customer zip'i ekle (order_id üzerinden)
        df = order_seller.merge(order_customer[["order_id", "customer_zip"]], on="order_id", how="left")

        # 5) Seller zip -> lat/lng
        df = df.merge(
            geo_zip.rename(columns={
                "geolocation_zip_code_prefix": "seller_zip",
                "geolocation_lat": "seller_lat",
                "geolocation_lng": "seller_lng",
            }),
            on="seller_zip",
            how="left"
        )

        # 6) Customer zip -> lat/lng
        df = df.merge(
            geo_zip.rename(columns={
                "geolocation_zip_code_prefix": "customer_zip",
                "geolocation_lat": "customer_lat",
                "geolocation_lng": "customer_lng",
            }),
            on="customer_zip",
            how="left"
        )

        # 7) Mesafe (km) — haversine_distance scalar bekler, bu yüzden satır satır apply
        df["distance_km"] = df.apply(
            lambda r: haversine_distance(
                r["seller_lng"], r["seller_lat"],
                r["customer_lng"], r["customer_lat"]
            ),
            axis=1
        )

        # 8) Sipariş başına ortalama seller-customer mesafesi
        out = (
            df.groupby("order_id")["distance_km"]
              .mean()
              .reset_index(name="distance_seller_customer")
        )

        return out

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_items', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        df = self.get_wait_time(is_delivered=is_delivered)

        df = df.merge(self.get_review_score(), on="order_id", how="left")
        df = df.merge(self.get_number_items(), on="order_id", how="left")
        df = df.merge(self.get_number_sellers(), on="order_id", how="left")
        df = df.merge(self.get_price_and_freight(), on="order_id", how="left")

        if with_distance_seller_customer:
            df = df.merge(self.get_distance_seller_customer(), on="order_id", how="left")

        # clean training set
        df = df.dropna()

        # kolon sırası (opsiyonel ama test bazen ister)
        cols = [
            "order_id",
            "wait_time",
            "expected_wait_time",
            "delay_vs_expected",
            "order_status",
            "dim_is_five_star",
            "dim_is_one_star",
            "review_score",
            "number_of_items",
            "number_of_sellers",
            "price",
            "freight_value",
        ]
        if with_distance_seller_customer:
            cols.append("distance_seller_customer")

        return df[cols]
