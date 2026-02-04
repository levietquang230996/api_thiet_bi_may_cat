import csv
import pandas as pd
import sys
import os
from collections import Counter

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'devicesPMISMayCat.csv')
output_file = os.path.join(script_dir, 'analysis_result.txt')

# Open output file
output = open(output_file, 'w', encoding='utf-8')

def print_to_file(*args, **kwargs):
    """Print to both file and console"""
    print(*args, **kwargs, file=output)
    try:
        print(*args, **kwargs)
    except:
        pass

# Read CSV with semicolon delimiter
df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')

print_to_file("=" * 80)
print_to_file("ĐÁNH GIÁ CHI TIẾT FILE: devicesPMISMayCat.csv")
print_to_file("=" * 80)

# 1. Thông tin cơ bản
print_to_file("\n1. THÔNG TIN CƠ BẢN:")
print_to_file(f"   - Tổng số dòng dữ liệu: {len(df):,}")
print_to_file(f"   - Tổng số cột: {len(df.columns)}")
print_to_file(f"   - Kích thước file: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# 2. Danh sách các cột
print_to_file("\n2. DANH SÁCH CÁC CỘT:")
for i, col in enumerate(df.columns, 1):
    print_to_file(f"   {i:2d}. {col}")

# 3. Kiểm tra dữ liệu thiếu
print_to_file("\n3. PHÂN TÍCH DỮ LIỆU THIẾU:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Cột': missing_data.index,
    'Số giá trị thiếu': missing_data.values,
    'Tỷ lệ (%)': missing_percent.values
})
missing_df = missing_df[missing_df['Số giá trị thiếu'] > 0].sort_values('Số giá trị thiếu', ascending=False)
if len(missing_df) > 0:
    print_to_file(missing_df.to_string(index=False))
else:
    print_to_file("   ✓ Không có dữ liệu thiếu")

# 4. Kiểm tra giá trị NULL
print_to_file("\n4. PHÂN TÍCH GIÁ TRỊ NULL:")
null_values = (df == 'NULL').sum()
null_df = pd.DataFrame({
    'Cột': null_values.index,
    'Số giá trị NULL': null_values.values,
    'Tỷ lệ (%)': (null_values / len(df) * 100).values
})
null_df = null_df[null_df['Số giá trị NULL'] > 0].sort_values('Số giá trị NULL', ascending=False)
if len(null_df) > 0:
    print_to_file(null_df.to_string(index=False))
else:
    print_to_file("   ✓ Không có giá trị NULL")

# 5. Phân tích các cột quan trọng
print_to_file("\n5. PHÂN TÍCH CÁC CỘT QUAN TRỌNG:")

# CATEGORYID
if 'CATEGORYID' in df.columns:
    print_to_file(f"\n   CATEGORYID:")
    print_to_file(f"   - Số giá trị duy nhất: {df['CATEGORYID'].nunique()}")
    print_to_file(f"   - Các giá trị: {df['CATEGORYID'].unique()}")

# P_MANUFACTURERID_DESC
if 'P_MANUFACTURERID_DESC' in df.columns:
    print_to_file(f"\n   P_MANUFACTURERID_DESC (Nhà sản xuất):")
    manufacturer_counts = df['P_MANUFACTURERID_DESC'].value_counts()
    print_to_file(f"   - Số nhà sản xuất: {df['P_MANUFACTURERID_DESC'].nunique()}")
    print_to_file(f"   - Top 10 nhà sản xuất:")
    for idx, (mfg, count) in enumerate(manufacturer_counts.head(10).items(), 1):
        print_to_file(f"     {idx:2d}. {mfg}: {count:,} thiết bị ({count/len(df)*100:.1f}%)")

# NATIONALFACT
if 'NATIONALFACT' in df.columns:
    print_to_file(f"\n   NATIONALFACT (Quốc gia sản xuất):")
    country_counts = df['NATIONALFACT'].value_counts()
    print_to_file(f"   - Số quốc gia: {df['NATIONALFACT'].nunique()}")
    for idx, (country, count) in enumerate(country_counts.items(), 1):
        print_to_file(f"     {idx}. {country}: {count:,} thiết bị ({count/len(df)*100:.1f}%)")

# DATEMANUFACTURE
if 'DATEMANUFACTURE' in df.columns:
    print_to_file(f"\n   DATEMANUFACTURE (Năm sản xuất):")
    df['DATEMANUFACTURE'] = pd.to_numeric(df['DATEMANUFACTURE'], errors='coerce')
    print_to_file(f"   - Năm sớm nhất: {df['DATEMANUFACTURE'].min():.0f}")
    print_to_file(f"   - Năm muộn nhất: {df['DATEMANUFACTURE'].max():.0f}")
    print_to_file(f"   - Năm trung bình: {df['DATEMANUFACTURE'].mean():.1f}")
    year_counts = df['DATEMANUFACTURE'].value_counts().sort_index()
    print_to_file(f"   - Phân bố theo năm (top 10 năm có nhiều thiết bị nhất):")
    for idx, (year, count) in enumerate(year_counts.tail(10).items(), 1):
        print_to_file(f"     {idx}. {year:.0f}: {count:,} thiết bị")

# OWNER_DESC
if 'OWNER_DESC' in df.columns:
    print_to_file(f"\n   OWNER_DESC (Chủ sở hữu):")
    owner_counts = df['OWNER_DESC'].value_counts()
    for idx, (owner, count) in enumerate(owner_counts.items(), 1):
        print_to_file(f"     {idx}. {owner}: {count:,} thiết bị ({count/len(df)*100:.1f}%)")

# LOAI_DESC
if 'LOAI_DESC' in df.columns:
    print_to_file(f"\n   LOAI_DESC (Loại):")
    loai_counts = df['LOAI_DESC'].value_counts()
    print_to_file(f"   - Số loại khác nhau: {df['LOAI_DESC'].nunique()}")
    if df['LOAI_DESC'].nunique() > 0:
        print_to_file(f"   - Top 10 loại:")
        for idx, (loai, count) in enumerate(loai_counts.head(10).items(), 1):
            print_to_file(f"     {idx:2d}. {loai}: {count:,} thiết bị")

# KIEU_MC_DESC
if 'KIEU_MC_DESC' in df.columns:
    print_to_file(f"\n   KIEU_MC_DESC (Kiểu máy cắt):")
    kieu_counts = df['KIEU_MC_DESC'].value_counts()
    print_to_file(f"   - Số kiểu khác nhau: {df['KIEU_MC_DESC'].nunique()}")
    if df['KIEU_MC_DESC'].nunique() > 0:
        print_to_file(f"   - Top 10 kiểu:")
        for idx, (kieu, count) in enumerate(kieu_counts.head(10).items(), 1):
            print_to_file(f"     {idx:2d}. {kieu}: {count:,} thiết bị")

# 6. Kiểm tra dữ liệu trùng lặp
print_to_file("\n6. KIỂM TRA DỮ LIỆU TRÙNG LẶP:")
duplicates = df.duplicated()
print_to_file(f"   - Số dòng trùng lặp hoàn toàn: {duplicates.sum()}")
if 'ASSETID' in df.columns:
    asset_duplicates = df['ASSETID'].duplicated()
    print_to_file(f"   - Số ASSETID trùng lặp: {asset_duplicates.sum()}")
    if asset_duplicates.sum() > 0:
        print_to_file(f"   - Các ASSETID trùng lặp:")
        dup_assets = df[df['ASSETID'].duplicated(keep=False)]['ASSETID'].value_counts()
        for asset, count in dup_assets.head(10).items():
            print_to_file(f"     • {asset}: {count} lần")

# 7. Thống kê tổng hợp
print_to_file("\n7. THỐNG KÊ TỔNG HỢP:")
print_to_file(f"   - Tổng số thiết bị: {len(df):,}")
print_to_file(f"   - Số thiết bị có đầy đủ thông tin (không có NULL): {len(df[~(df == 'NULL').any(axis=1)]):,}")
print_to_file(f"   - Số thiết bị thiếu thông tin: {len(df[(df == 'NULL').any(axis=1)]):,}")

# 8. Kiểm tra tính nhất quán dữ liệu
print_to_file("\n8. KIỂM TRA TÍNH NHẤT QUÁN:")
if 'ASSETID' in df.columns and 'ASSETDESC' in df.columns:
    empty_desc = df[df['ASSETDESC'].isna() | (df['ASSETDESC'] == 'NULL')]
    print_to_file(f"   - Số thiết bị không có mô tả: {len(empty_desc)}")

print_to_file("\n" + "=" * 80)
print_to_file("HOÀN TẤT ĐÁNH GIÁ")
print_to_file("=" * 80)

output.close()
print(f"\nKết quả đã được lưu vào: {output_file}")
