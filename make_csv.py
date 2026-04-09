import pandas as pd

df = pd.read_csv("D:/SentinelNet/data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

df = df.drop(columns=['Label'], errors='ignore')

df.to_csv("test_data.csv", index=False)

print("✅ Full dataset saved")