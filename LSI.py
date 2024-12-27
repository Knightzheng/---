import json
import jieba
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import dok_matrix, csr_matrix
import gc
import os
from matplotlib import font_manager
from adjustText import adjust_text  # 用于优化标签位置
import random
import re
from sklearn.preprocessing import StandardScaler
import warnings  # 引入warnings模块
#1. 设置中文字体
def set_chinese_font():
    #设置 Matplotlib 使用中文字体。如 SimHei 或 Microsoft YaHei。
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    if 'SimHei' not in available_fonts:
        plt.rcParams['font.family'] = ['Microsoft YaHei']
        if 'Microsoft YaHei' not in available_fonts:
            font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
            else:
                print("未找到支持中文的字体，请确保系统中安装了 SimHei 或 Microsoft YaHei 字体。")
    return
#2. 数据加载与预处理
def load_stopwords(filepath):
    #加载停用词列表。
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    print(f"加载的停用词总数: {len(stopwords)}")
    return stopwords
def load_news_data(filepath):
    # 加载并解析每行一个 JSON 对象的新闻数据。
    news_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    news_item = json.loads(line)
                    news_data.append(news_item)
                except json.JSONDecodeError as e:
                    print(f"JSON解码错误: {e} 在第 {line_number} 行: {line}")
    print(f"加载的文档总数: {len(news_data)}")
    return news_data
def tokenize_documents(news_data, stopwords):
    # 对新闻内容进行分词并去除停用词。
    tokenized_docs = []
    word_counter = Counter()

    for doc in news_data:
        content = doc.get('content', '')
        tokens = jieba.lcut(content)
        tokens = [token for token in tokens if token.strip() and token not in stopwords]
        tokenized_docs.append(tokens)
        word_counter.update(tokens)

    print(f"完成分词，总文档数: {len(tokenized_docs)}")
    return tokenized_docs, word_counter
def build_vocabulary(word_counter, vocab_size=30000):
    #根据词频限制词汇表大小。返回词汇表、词汇索引及最常见的词项列表。
    most_common_words = word_counter.most_common(vocab_size)
    vocab = [word for word, freq in most_common_words]
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    print(f"限制后的词汇表大小: {len(vocab)}")
    return vocab, vocab_index, most_common_words
def build_cooccurrence_matrix(tokenized_docs, vocab_index, window_size=5):
    #构建稀疏词项-词项共现矩阵。
    cooc_dict = defaultdict(lambda: defaultdict(int))

    for tokens in tokenized_docs:
        token_length = len(tokens)
        for i, token in enumerate(tokens):
            if token not in vocab_index:
                continue
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, token_length)
            for j in range(start, end):
                if i == j:
                    continue
                context_token = tokens[j]
                if context_token not in vocab_index:
                    continue
                cooc_dict[token][context_token] += 1

    print("完成词项共现计数")

    # 使用 DOK 稀疏矩阵构建词项-词项共现矩阵
    size = len(vocab_index)
    cooc_matrix = dok_matrix((size, size), dtype=np.float32)

    for word, contexts in cooc_dict.items():
        i = vocab_index[word]
        for context, count in contexts.items():
            j = vocab_index[context]
            cooc_matrix[i, j] = count

    # 释放不再需要的内存
    del cooc_dict
    gc.collect()

    # 转换为CSR格式以提高计算效率
    cooc_matrix = cooc_matrix.tocsr()
    print(f"词项共现矩阵形状: {cooc_matrix.shape}")

    return cooc_matrix
def build_term_document_matrix(tokenized_docs, vocab):
    #构建词项-文档矩阵。
    documents = [' '.join(doc) for doc in tokenized_docs]
    vectorizer = CountVectorizer(vocabulary=vocab, binary=False, lowercase=False)  # 设置 lowercase=False
    term_doc_matrix = vectorizer.fit_transform(documents)
    print(f"词项-文档矩阵形状: {term_doc_matrix.shape}")
    return term_doc_matrix
#3. SVD降维
def perform_svd(matrix, k):
    """执行奇异值分解（SVD）。
    参数:
    - matrix (scipy.sparse.csr_matrix): 输入矩阵。
    - k (int): 要保留的潜在维度数。
    返回:
    - svd (TruncatedSVD): SVD模型。
    - U (numpy.ndarray): 词项的潜在向量。
    - V (numpy.ndarray): 文档的潜在向量。
    - explained_variance (float): 解释的方差比例。"""
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(matrix)  # 词项的潜在向量 (terms x k)
    Vt = svd.components_  # V^T (k x documents)
    V = Vt.T * svd.singular_values_  # 文档的潜在向量 (documents x k)
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"SVD 处理 k={k}: 解释的方差比例: {explained_variance:.4f}")
    return svd, U, V, explained_variance
def standardize_vectors(U, V):
    """标准化词项和文档向量。
    参数:
    - U (numpy.ndarray): 词项的潜在向量。
    - V (numpy.ndarray): 文档的潜在向量。
    返回:
    - U_scaled (numpy.ndarray): 标准化后的词项向量。
    - V_scaled (numpy.ndarray): 标准化后的文档向量。"""
    scaler = StandardScaler()
    U_scaled = scaler.fit_transform(U)
    V_scaled = scaler.fit_transform(V)
    return U_scaled, V_scaled
def merge_vectors(U_scaled, V_scaled):
    """合并词项和文档的标准化向量。
    参数:
    - U_scaled (numpy.ndarray): 标准化后的词项向量。
    - V_scaled (numpy.ndarray): 标准化后的文档向量。
    返回:
    - combined_matrix (numpy.ndarray): 合并后的向量矩阵。"""
    combined_matrix = np.vstack((U_scaled, V_scaled))
    return combined_matrix
#4. 降维与可视化
def apply_tsne(combined_matrix, random_seed=42, perplexity=50, learning_rate=300, max_iter=2000):
    """对合并后的向量应用 t-SNE 降维到2D空间。
    参数:
    - combined_matrix (numpy.ndarray): 合并后的向量矩阵。
    - random_seed (int): 随机种子，确保结果可重复。
    - perplexity (int): t-SNE的perplexity参数。
    - learning_rate (int): t-SNE的学习率参数。
    - n_iter (int): t-SNE的迭代次数。
    返回:
    - tsne_results (numpy.ndarray): 降维后的2D向量。"""
    tsne = TSNE(
        n_components=2,
        random_state=random_seed,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        verbose=0  # 关闭详细输出
    )
    tsne_results = tsne.fit_transform(combined_matrix)
    return tsne_results
def is_chinese(word):
    #如果词完全由中文字符组成，返回True，否则返回False。
    return re.match(r'^[\u4e00-\u9fff]+$', word) is not None
def visualize_combined_subset_tsne(tsne_results, vocab, sampled_terms_indices, sampled_docs_indices, k, save_dir='plots'):
    """
    可视化词项和文档在同一2D空间中的分布，仅显示随机选择的2000个词项和1000个文档。
    参数:
    - tsne_results (numpy.ndarray): 降维后的2D向量。
    - vocab (List[str]): 词汇表。
    - sampled_terms_indices (List[int]): 采样的词项索引。
    - sampled_docs_indices (List[int]): 采样的文档索引。
    - k (int): 当前的k值。
    - save_dir (str): 图像保存目录。
    """
    # 获取词项和文档的数量
    num_terms = len(vocab)
    num_docs = tsne_results.shape[0] - num_terms

    # 计算文档在tsne_results中的实际索引
    # 将 sampled_docs_indices 每个元素加上 num_terms
    docs_indices = [num_terms + idx for idx in sampled_docs_indices]

    # 获取采样词项和文档的t-SNE坐标
    terms_tsne = tsne_results[sampled_terms_indices]
    docs_tsne = tsne_results[docs_indices]

    plt.figure(figsize=(16, 12))

    # 绘制文档点
    plt.scatter(docs_tsne[:, 0], docs_tsne[:, 1], c='blue', alpha=0.5, label='文档', s=10)

    # 绘制词项点
    plt.scatter(terms_tsne[:, 0], terms_tsne[:, 1], c='red', alpha=0.5, label='词项', s=10)

    # 随机选择部分词项和文档进行标注
    # 为避免图像过于杂乱，标注的数量可以适当减少
    num_term_labels = min(50, len(sampled_terms_indices))
    num_doc_labels = min(50, len(sampled_docs_indices))

    term_labels_indices = random.sample(range(len(sampled_terms_indices)), num_term_labels)
    doc_labels_indices = random.sample(range(len(sampled_docs_indices)), num_doc_labels)

    # 标注词项
    for idx in term_labels_indices:
        word = vocab[sampled_terms_indices[idx]]
        plt.text(terms_tsne[idx, 0], terms_tsne[idx, 1], word, fontsize=9, alpha=0.7)

    # 标注文档（可以用文档编号代替具体内容）
    for idx in doc_labels_indices:
        doc_num = sampled_docs_indices[idx]
        plt.text(docs_tsne[idx, 0], docs_tsne[idx, 1], f'Doc {doc_num}', fontsize=9, alpha=0.7)

    plt.title(f'随机2000词项和200文档的 t-SNE 可视化 (k={k})')
    plt.legend()
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, f'combined_subset_tsne_k_{k}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"组合子集图像已保存至 {save_path}")
    plt.close()
def visualize_tsne_random(matrix, labels, title, num_labels=200, save_path=None, random_seed=42):
    """
    使用 t-SNE 可视化降维后的矩阵，并随机标注指定数量的中文标签。
    参数:
    - matrix (numpy.ndarray): 降维后的词项矩阵。
    - labels (List[str]): 词项标签列表。
    - title (str): 图像标题。
    - num_labels (int): 要标注的词项数量。
    - save_path (str): 图像保存路径。如果为None，则显示图像。
    - random_seed (int): 随机种子，确保结果可重复。
    """
    # 设置随机种子以确保可重复性
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 筛选出中文词项的索引
    chinese_indices = [i for i, label in enumerate(labels) if is_chinese(label)]
    total_chinese = len(chinese_indices)
    print(f"总中文词项数: {total_chinese}")

    if total_chinese == 0:
        print("没有找到中文词项，无法标注。")
        return

    # 确定要标注的词项数量
    if num_labels > total_chinese:
        num_labels = total_chinese
        print(f"警告：num_labels 大于中文词项总数。将标注所有 {total_chinese} 个中文词项。")

    # 从中文词项中随机选择要标注的索引
    selected_indices = random.sample(chinese_indices, num_labels)
    selected_labels = [labels[i] for i in selected_indices]

    # 执行 t-SNE
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30, max_iter=2000, verbose=0)  # 设置 verbose=0
    tsne_results = tsne.fit_transform(matrix)

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], legend=False, alpha=0.6)

    # 标注随机选择的中文词项
    texts = []
    for i, label in zip(selected_indices, selected_labels):
        texts.append(plt.text(tsne_results[i, 0], tsne_results[i, 1], label, fontsize=9, alpha=0.7))

    # 优化标签位置，避免重叠
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至 {save_path}")
    else:
        plt.show()
    plt.close()
def visualize_tsne_documents(matrix, title, sample_size=200, perplexity=50, learning_rate=300, max_iter=2000, save_path=None):
    """
    使用 t-SNE 可视化降维后的文档矩阵，并保存图像。
    参数:
    - matrix (numpy.ndarray): 降维后的文档矩阵。
    - title (str): 图像标题。
    - sample_size (int): 要采样的文档数量。
    - perplexity (int): t-SNE的perplexity参数。
    - learning_rate (int): t-SNE的学习率参数。
    - n_iter (int): t-SNE的迭代次数。
    - save_path (str): 图像保存路径。如果为None，则显示图像。
    """
    if sample_size < matrix.shape[0]:
        indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
        sampled_matrix = matrix[indices]
    else:
        sampled_matrix = matrix

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        verbose=0  # 关闭详细输出
    )
    tsne_results = tsne.fit_transform(sampled_matrix)

    plt.figure(figsize=(14, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6, c='green', label='文档')

    # 增加文档编号标注
    for i in range(len(tsne_results)):
        plt.text(tsne_results[i, 0], tsne_results[i, 1], f'Doc {i}', fontsize=9, alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"文档可视化图像已保存至 {save_path}")
    else:
        plt.show()
    plt.close()
#5. 综合分析与可视化
def analyze_lsi(cooc_matrix, term_doc_matrix, vocab, most_common_words, k_values, save_dir='plots'):
    """
    分析不同 k 值对 LSI 的影响，并生成可视化图像。
    同时将词项和文档投影到同一2D空间中进行可视化，仅显示随机的2000个词项和200个文档。
    参数:
    - cooc_matrix (scipy.sparse.csr_matrix): 词项-词项共现矩阵。
    - term_doc_matrix (scipy.sparse.csr_matrix): 词项-文档矩阵。
    - vocab (List[str]): 词汇表。
    - most_common_words (List[Tuple[str, int]]): 词频最高的词项及其频率。
    - k_values (List[int]): 要尝试的k值列表。
    - save_dir (str): 图像保存目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    explained_variances = []
    document_vectors_dict = {}  # 存储不同k值下的文档向量

    for k in k_values:
        print(f"\n处理 k={k}")

        # 对词项-词项矩阵执行 SVD
        svd_cooc, U_cooc, V_cooc, ev_cooc = perform_svd(cooc_matrix, k)
        print(f"词项-词项 SVD: 解释的方差比例={ev_cooc:.4f}")

        # 对词项-文档矩阵执行 SVD
        svd_term_doc, U_term_doc, V_term_doc, ev_term_doc = perform_svd(term_doc_matrix, k)
        print(f"词项-文档 SVD: 解释的方差比例={ev_term_doc:.4f}")

        explained_variances.append((k, ev_cooc, ev_term_doc))

        # 标准化向量
        U_scaled_cooc, V_scaled_cooc = standardize_vectors(U_cooc, V_cooc)
        U_scaled_term_doc, V_scaled_term_doc = standardize_vectors(U_term_doc, V_term_doc)

        # 合并词项和文档的向量
        combined_matrix = np.vstack((U_scaled_cooc, V_scaled_term_doc))

        # 应用 t-SNE
        tsne_results = apply_tsne(combined_matrix, random_seed=42, perplexity=30, learning_rate=200, max_iter=1000)

        # 可视化词项和文档的组合子集t-SNE
        visualize_combined_subset_tsne(
            tsne_results,
            vocab,
            sampled_terms_indices=random.sample(range(len(vocab)), 2000),
            sampled_docs_indices=random.sample(range(V_term_doc.shape[0]), 1000),
            k=k,
            save_dir=save_dir
        )
        # 存储文档向量以便单独降维和可视化
        document_vectors = V_term_doc
        document_vectors_dict[k] = document_vectors

        # 单独对文档向量进行降维和可视化
        # 这里可以调整t-SNE参数，如perplexity和迭代次数
        tsne_save_path_doc = os.path.join(save_dir, f'document_tsne_k_{k}.png')
        visualize_tsne_documents(
            document_vectors,
            f'文档的 t-SNE 可视化 (k={k})',
            sample_size=200,
            perplexity=30,  # 可尝试调整
            learning_rate=200,  # 可尝试调整
            max_iter=1000,  # 可尝试增加
            save_path=tsne_save_path_doc
        )

        # 生成其他可视化（保留之前的功能）
        # 1. 可视化词项 - 全量词项（随机标注100个中文词）
        all_words = [word for word, freq in most_common_words]
        tsne_save_path_full = os.path.join(save_dir, f'word_tsne_full_k_{k}.png')
        visualize_tsne_random(
            U_cooc,
            all_words,
            f'词项的 t-SNE 可视化 (k={k}) - 全量',
            num_labels=100,
            save_path=tsne_save_path_full,
            random_seed=42
        )

        # 2. 可视化词项 - 随机选择的2000个中文词（随机标注100个中文词）
        # 首先，确保词汇表中至少有2000个中文词
        chinese_words = [word for word, freq in most_common_words if is_chinese(word)]
        if len(chinese_words) < 2000:
            print(f"警告：中文词汇表大小为 {len(chinese_words)}，不足2000个词。将标注所有中文词。")
            random_indices_2000 = [i for i, word in enumerate(all_words) if is_chinese(word)]
        else:
            # 随机选择2000个中文词
            chinese_indices_all = [i for i, word in enumerate(all_words) if is_chinese(word)]
            random_indices_2000 = random.sample(chinese_indices_all, 2000)

        reduced_cooc_random2000 = U_cooc[random_indices_2000]
        tsne_save_path_random2000 = os.path.join(save_dir, f'word_tsne_random2000_k_{k}.png')
        visualize_tsne_random(
            reduced_cooc_random2000,
            [all_words[i] for i in random_indices_2000],
            f'词项的 t-SNE 可视化 (k={k}) - 随机2000词',
            num_labels=min(100, len(random_indices_2000)),
            save_path=tsne_save_path_random2000,
            random_seed=42
        )
    # ----------------------------- 6. 主函数 -----------------------------
def main():
    # 忽略特定的警告
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    # 设置中文字体
    set_chinese_font()
    # 文件路径（根据实际情况修改）
    stopwords_path = 'data/news/stopwords-zh.txt'
    news_data_path = 'data/news/news2016zh_valid_2000.json'
    save_directory = 'tsne_visualizations/LSI'  # 生成的图像将保存在此目录
    # 1. 加载停用词
    stopwords = load_stopwords(stopwords_path)
    # 2. 加载新闻数据
    news_data = load_news_data(news_data_path)
    # 3. 分词并统计词频
    tokenized_docs, word_counter = tokenize_documents(news_data, stopwords)
    # 4. 限制词汇表大小
    vocab, vocab_index, most_common_words = build_vocabulary(word_counter, vocab_size=30000)
    # 5. 构建词项-词项共现矩阵
    cooc_matrix = build_cooccurrence_matrix(tokenized_docs, vocab_index, window_size=5)
    # 6. 构建词项-文档矩阵
    term_doc_matrix = build_term_document_matrix(tokenized_docs, vocab)
    # 7. 定义 k 值
    k_values = [100, 200, 300]
    # 8. 执行分析并生成可视化图像
    analyze_lsi(cooc_matrix, term_doc_matrix, vocab, most_common_words, k_values, save_dir=save_directory)
if __name__ == '__main__':
    main()
