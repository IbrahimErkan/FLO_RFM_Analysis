# FLO-CUSTOMER SEGMENTATİON WİTH RFM


# 1. Data Understanding
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%3.f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


# 2. Data Preparation

def check_df(df, head=10):
    print("-------------------- Shape --------------------")
    print(df.shape)
    print("\n-------------------- Types --------------------")
    print(df.dtypes)
    print("\n-------------------- Head --------------------")
    print(df.head(head))
    print("\n-------------------- Tail --------------------")
    print(df.tail(head))
    print("\n-------------------- Na --------------------")
    print(df.isnull().sum())
    print("\n-------------------- Quantiles --------------------")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("\n-------------------- Features Names --------------------")
    print(df.columns)


check_df(df)

df.info()

df.nunique()

df["order_channel"].value_counts()

df["interested_in_categories_12"].value_counts()

# Veride ayrı ayrı olan offline ve online toplam alışverişi, total_price başlığı altında gösteriyoruz.
# Aynı şekilde offline ve online olarak ayrı ayrı gösterilmiş sipariş adetlerini, yani toplam harcamasını ise
# total_expenditure başlığı altına alıyoruz.
df["total_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["total_expenditure"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df.head()

df.info()

#1st way:
date_convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[date_convert] = df[date_convert].apply(pd.to_datetime)
#2nd way:
# date_convert = df.columns[df.columns.str.contains("date")]
# df[date_convert] = df[date_convert].apply(pd.to_datetime)

df.groupby("order_channel").agg({"total_of_purchases" : "sum"}).head()

df.groupby("order_channel").agg({"total_of_purchases" : "sum",
                                'total_expenditure':'sum',
                                'order_channel' : 'count'}).sort_values(by = "total_expenditure", ascending=False).head()

df.sort_values("total_expenditure", ascending=False)[:10]

df.groupby("master_id").agg({"total_of_purchases" : "sum"
                            }).sort_values(by = "total_of_purchases", ascending=False).head(10)


# Veri ön hazırlık sürecini fonksiyonlaştırma.

def data_(dataframe):
    dataframe["total_of_purchases"] = dataframe["order_num_total_ever_online"] + dataframe[
        "order_num_total_ever_offline"]
    dataframe["total_expenditure"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]

    date_convert = dataframe.columns[dataframe.columns.str.contains("date")]

    dataframe[date_convert] = dataframe[date_convert].apply(pd.to_datetime)

    return dataframe

data_(df)


# 4. Calculating RFM Metrics

# Recency, Frequency ve Monetary tanımları?

# Recency(yenilik) : "En son ne zaman alışveriş yaptı" durumunu ifade eder.
# Frequency(sıklık) : Müşterinin yaptığı toplam alışveriş sayısıdır.
# Monetary(Parasal Değer) : Müşterilerimizin bize kazandırdığı toplam kazanç ifade eder.

df["last_order_date"].max()   # Timestamp('2021-05-30 00:00:00')
t_date = dt.datetime(2021, 6, 1)

# 1st way:
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (t_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["total_of_purchases"]
rfm["monetary"] = df["total_expenditure"]

rfm.head()

# 2nd way:
# rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (t_date - last_order_date.max()).days,
#                                  'total_of_purchases': lambda total_of_purchases: total_of_purchases.sum(),
#                                  'total_expenditure': lambda total_expenditure: total_expenditure.sum()})


# 5. Calculating RFM Scores

# Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1–5 arasında skorlara çeviriyoruz.
# Recency değerinin düşük olması daha fazla puan edeceğinden 5'ten 1'e,
# frequency ve monetary değerlerinin yüksek olması daha fazla skor sağlayacağından 1'den 5'e sıralıyoruz.

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])   # 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

pd.crosstab(rfm["frequency"], rfm["frequency_score"]).head(8)
len(rfm)

rfm["frequency_score"].value_counts()

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()

rfm[rfm["RFM_SCORE"] == "51"].head(10)


# 6. Creating & Analysing RFM Segments

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
seg_map

rfm['Segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.head()

Srfm = ["Segment", "recency", "frequency", "monetary"]
rfm[Srfm].groupby("Segment").agg(["mean", "count"])

rfm[rfm["Segment"] == "champions"].head()

Segments = rfm['Segment'].value_counts().sort_values(ascending=False)
Segments


rfm.groupby("Segment").agg({'recency' : 'mean',
                            'frequency' : 'mean',
                            'monetary' : 'mean'}).sort_values(by = 'monetary', ascending = False)


# Case 1:
# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçmek isteniliyor.
# Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

Seg_A = rfm[(rfm["Segment"]=="champions") | (rfm["Segment"]=="loyal_customers")]["customer_id"]
Seg_A

Seg_B = df[(df["master_id"].isin(Seg_A)) & (df["interested_in_categories_12"]).str.contains("KADIN")]["master_id"]
Seg_B.shape[0]

Seg_B.to_csv("first_case_new_customers.csv")


# Case 2:
# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

Seg_C = rfm[(rfm["Segment"] == "cant_loose") | (rfm["Segment"] == "about_to_sleep") | (rfm["Segment"] == "new_customers")]["customer_id"]
Seg_C.head()

Seg_D = df[(df["master_id"].isin(Seg_C)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
Seg_D.head()

Seg_D.to_csv("second_case_new_customers.csv")