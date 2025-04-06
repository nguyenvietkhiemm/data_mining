import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import io

# tóm tắt df
def df_info(target_df, topN=10):    
    max_row = target_df.shape[0]
    print(f'Shape: {target_df.shape}')
    
    df = target_df.dtypes.to_frame()
    df.columns = ['DataType']
    df['#Nulls'] = target_df.isnull().sum()
    df['#Uniques'] = target_df.nunique()
    
    # thống kê
    df['Min']   = target_df.min(numeric_only=True)
    df['Mean']  = target_df.mean(numeric_only=True)
    df['Std']   = target_df.std(numeric_only=True)
    df['Max']   = target_df.max(numeric_only=True)
    
    # top 10
    df[f'top{topN} val'] = None
    df[f'top{topN} cnt'] = None
    df[f'top{topN} ratio'] = None
    for c in df.index:
        vc = target_df[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        ratio = list((vc.values / max_row).round(3))
        df.loc[c, f'top{topN} val'] = ', '.join(map(str, val))
        df.loc[c, f'top{topN} cnt'] = ', '.join(map(str, cnt))
        df.loc[c, f'top{topN} ratio'] = ', '.join(map(str, ratio))

    return df

# biểu đồ venn so sánh giá trị giữa 2 tập
def venn_diagram(train, test, category_features, names=('train', 'test')):
    num_features = len(category_features)
    cols = 3  
    rows = int(np.ceil(num_features / cols))
    figsize=(20, 4 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    axes = axes.flatten() if num_features > 1 else [axes]
    
    for i, c in enumerate(category_features):
        venn2([set(train[c].unique()), set(test[c].unique())], set_labels=names, ax=axes[i])
        axes[i].set_title(f'{c}', fontsize=16)

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    
    return buf

# biểu đồ tần suất feature rời rạc
def count_categories(df, category_features, topN=30, sort='freq', df2=None, description=None):
    num_features = len(category_features)
    cols = 2

    fig, axes = plt.subplots(num_features, cols, figsize=(20, 8 * num_features))

    for i, c in enumerate(category_features):
        target_value = df[c].value_counts().head(topN).index
        order = target_value if sort == 'freq' else df[c].value_counts().head(topN).sort_index().index

        sns.countplot(x=c, data=df[df[c].isin(order)], order=order, ax=axes[i, 0])
        axes[i, 0].set_title(f'{c} - Dataset 1')
        axes[i, 0].tick_params(axis='x', rotation=90)
        for container in axes[i, 0].containers:
            axes[i, 0].bar_label(container, fmt='%d')
            
        if description:
            axes[i, 0].set_xlabel(description[i]) # chỗ này có bug tiềm ẩn
        
        sns.countplot(x=c, data=df2[df2[c].isin(order)], order=order, ax=axes[i, 1])
        axes[i, 1].set_title(f'{c} - Dataset 2')
        axes[i, 1].tick_params(axis='x', rotation=90)
        axes[i, 1].set_xlabel('')
        for container in axes[i, 1].containers:
            axes[i, 1].bar_label(container, fmt='%d')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

    return buf

# histogram giá trị liên tục
def hist_continuous(df, continuous_features, bins=30, df2=None, description=None):
    num_features = len(continuous_features)
    cols = 2

    fig, axes = plt.subplots(num_features, cols, figsize=(20, 8*num_features))

    for i, c in enumerate(continuous_features):
        df[c].hist(bins=bins, ax=axes[i, 0])
        axes[i, 0].set_title(f'{c} - Dataset 1')
        for container in axes[i, 0].containers:
            axes[i, 0].bar_label(container, fmt='%d')
            
        if description:
            axes[i, 0].set_xlabel(description[i]) # chỗ này có bug tiềm ẩn

        df2[c].hist(bins=bins, ax=axes[i, 1])
        axes[i, 1].set_title(f'{c} - Dataset 2')
        for container in axes[i, 1].containers:
            axes[i, 1].bar_label(container, fmt='%d')
        axes[i, 1].set_xlabel('')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

    return buf

# kiểm tra outliers
def detect_outliers(df, method="iqr", threshold=1.5):
    outliers = {}

    for col in df.select_dtypes(include=[np.number]):
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))
        
        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            mask = (np.abs((df[col] - mean) / std) > threshold)
        
        outliers[col] = df[col][mask].tolist()
    
    return outliers
