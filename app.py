from flask import Flask, request, jsonify, render_template
import numpy as np
import plotly.graph_objs as go
import plotly
import json
import numpy as np
import glob
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from concurrent.futures import ThreadPoolExecutor
from plotly.subplots import make_subplots
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd

app = Flask(__name__)

# 读取pkl文件
with open("eeg_max_min.pkl", "rb") as file:
    eeg_max_min = pickle.load(file)


def file_scanf(file_path):
    return np.array(glob.glob(f"/data1/share_data/purdue/{file_path}/*.pkl"))


class EEGDataset(Dataset):
    def __init__(self, paths):
        self.filepaths = paths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with open(self.filepaths[idx], "rb") as f:
            x = pickle.load(f)
            y = pickle.load(f)
        return x, y


s1_raw_dataset = EEGDataset(file_scanf("s1/raw"))
s1_trail_norm_dataset = EEGDataset(file_scanf("s1/trail_norm"))
s1_time_norm_dataset = EEGDataset(file_scanf("s1/time_norm"))
s1_robust_norm_dataset = EEGDataset(file_scanf("s1/robust_norm"))
s1_robust_time_norm_dataset = EEGDataset(file_scanf("s1/robust_time_norm"))
s1_01_trail_norm_dataset = EEGDataset(file_scanf("s1/01_trail_norm"))
gmm_trail_norm = EEGDataset(file_scanf("s1/gmm_trail_norm"))
gmm_time_norm = EEGDataset(file_scanf("s1/gmm_time_norm"))
full_raw_dataset = EEGDataset(file_scanf("raw"))
datasets = {
    "raw_s1": s1_raw_dataset,
    "trail_s1": s1_trail_norm_dataset,
    "time_s1": s1_time_norm_dataset,
    "robust_norm": s1_robust_norm_dataset,
    "robust_time_norm": s1_robust_time_norm_dataset,
    "s1_01_trail_norm": s1_01_trail_norm_dataset,
    "gmm_trail_norm": gmm_trail_norm,
    "gmm_time_norm": gmm_time_norm,
    "raw_full": full_raw_dataset,
}
colors = {
    "raw_s1": "blue",
    "trail_s1": "red",
    "time_s1": "green",
    "robust_norm": "purple",
    "robust_time_norm": "orange",
    "s1_01_trail_norm": "hotpink",
    "gmm_trail_norm": "brown",
    "gmm_time_norm": "gray",
    "raw_full": "hotpink",
}

title = {
    "raw_s1": "r",
    "trail_s1": "t",
    "time_s1": "c",
    "robust_norm": "Robust",
    "robust_time_norm": "Robust Time",
    "s1_01_trail_norm": "01 Trail",
    "gmm_time_norm": "GMM Time",
    "gmm_trail_norm": "GMM Trail",
    "raw_full": "f",
}

range_key = {
    "raw_s1": "raw",
    "trail_s1": "trail_norm",
    "time_s1": "time_norm",
    "raw_full": "raw_full",
}


seq_num = 8
subplot_height = 330




def reorder_similarity_matrix(sim_mat):
    """
    Reorder a similarity matrix based on hierarchical clustering.

    Args:
    sim_mat (np.array): A square array representing the similarity matrix.

    Returns:
    tuple: A tuple containing:
        - np.array: The reordered similarity matrix.
        - list: The order of indices used to reorder the matrix.
    """

    # Convert similarity matrix to a distance matrix
    distance_mat = 1 - sim_mat
    np.fill_diagonal(distance_mat, 0)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(spd.squareform(distance_mat), method="average")

    # Get the order of indices from the dendrogram
    order = sch.dendrogram(linkage_matrix, no_plot=True)["leaves"]

    # Reorder the similarity matrix according to the clustering result
    reordered_mat = sim_mat[np.ix_(order, order)]
    # print("Order of Indices:", order)
    return reordered_mat


def build_response(results, label):
    fig = go.Figure()
    channel_range_fig = go.Figure()
    sample_range_fig = go.Figure()
    similarity_matrix_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["原相似矩阵", "聚类重排矩阵"],
        horizontal_spacing=0.02,  # 设置子图之间的水平间距
    )
    sim_mat_dis_fig = go.Figure()
    sim_mat_seq_fig = make_subplots(
        rows=1,
        cols=seq_num,
        subplot_titles=[f"第 {i+1} 段" for i in range(seq_num)],
        horizontal_spacing=0.01,  # 设置子图之间的水平间距
        # vertical_spacing=0.1,  # 设置子图之间的垂直间距
    )
    sample_dis_fig = go.Figure()
    channel_dis_fig = go.Figure()

    for result in results:
        fig.add_traces(result["fig"])
        channel_range_fig.add_traces(result["channel_range_fig"])
        sample_range_fig.add_traces(result["sample_range_fig"])
        for i in range(2):
            similarity_matrix_fig.add_trace(
                result["similarity_matrix_fig"][i], row=1, col=i + 1
            )

        for i in range(seq_num):
            row = 1
            col = i + 1
            sim_mat_seq_fig.add_trace(result["sim_mat_seq_fig"][i], row=row, col=col)

        sim_mat_dis_fig.add_traces(result["sim_mat_dis_fig"])
        sample_dis_fig.add_traces(result["sample_dis_fig"])
        channel_dis_fig.add_traces(result["channel_dis_fig"])

    fig.update_layout(
        title="样本通道数值",
        xaxis_title="Time",
        yaxis_title="Value",
        title_x=0.5,  # 设置标题居中
    )

    channel_range_fig.update_layout(
        title="通道最值范围",
        xaxis_title="Channel",
        yaxis_title="Value",
        title_x=0.5,  # 设置标题居中
    )
    sample_range_fig.update_layout(
        title="全部样本最值范围",
        xaxis_title="Trail",
        yaxis_title="Value",
        title_x=0.5,  # 设置标题居中
    )
    similarity_matrix_fig.update_layout(
        title="通道相似度矩阵",
        title_x=0.5,  # 设置标题居中
    )

    sim_mat_seq_fig.update_layout(
        title="分段相似度矩阵",
        title_x=0.5,  # 设置标题居中
        showlegend=False,
        height=subplot_height,  # 设置子图高度
        # width=total_width,  # 根据子图数量和尺寸设置宽度
        # height=total_height,  # 设置高度，保持所有子图为正方形
        # margin=dict(l=20, r=20, t=90, b=20),  # 调整边距，确保标题和子图之间有足够空间
    )
    for i in range(seq_num):
        sim_mat_seq_fig.update_yaxes(scaleanchor="x" + str(i + 1), scaleratio=1)
    sim_mat_dis_fig.update_layout(
        title="通道相似度分布",
        title_x=0.5,  # 设置标题居中
    )

    sample_dis_fig.update_layout(
        title="样本数值分布",
        title_x=0.5,  # 设置标题居中
    )

    channel_dis_fig.update_layout(
        title="通道数值分布",
        title_x=0.5,  # 设置标题居中
    )
    response = {
        "label": json.dumps(label),
        "data": json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
        "layout": json.loads(
            json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "channel_dis_data": json.loads(
            json.dumps(channel_dis_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "channel_dis_layout": json.loads(
            json.dumps(channel_dis_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "channel_range_data": json.loads(
            json.dumps(channel_range_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "channel_range_layout": json.loads(
            json.dumps(channel_range_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "trail_range_data": json.loads(
            json.dumps(sample_range_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "trail_range_layout": json.loads(
            json.dumps(sample_range_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "similarity_matrix_data": json.loads(
            json.dumps(similarity_matrix_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "similarity_matrix_layout": json.loads(
            json.dumps(similarity_matrix_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "sim_mat_seq_data": json.loads(
            json.dumps(sim_mat_seq_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "sim_mat_seq_layout": json.loads(
            json.dumps(sim_mat_seq_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "sim_mat_dis_data": json.loads(
            json.dumps(sim_mat_dis_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "sim_mat_dis_layout": json.loads(
            json.dumps(sim_mat_dis_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "sample_dis_data": json.loads(
            json.dumps(sample_dis_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "sample_dis_layout": json.loads(
            json.dumps(sample_dis_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        ),
    }

    return jsonify(response)


def create_figures(row, name, color, channel, datasets):

    data = datasets[name][row][0]
    if "full" in name:
        data = data[:, ::4] * 1e5
    channel_data = data[channel]
    time_len = np.arange(len(channel_data))
    channel_len = np.arange(96)
    trail_len = np.arange(40000)
    fig_trace = go.Scatter(
        x=time_len,
        y=channel_data,
        mode="lines",
        name=title[name],
        line=dict(color=color),
    )

    channel_dis_trace = go.Histogram(
        x=channel_data,
        nbinsx=100,
        histnorm="probability density",
        marker_color=color,
        showlegend=False,
    )

    channel_range_values = np.max(data, axis=1) - np.min(data, axis=1)

    channel_range_trace = go.Scatter(
        x=channel_len,
        y=channel_range_values,
        mode="lines",
        name=f"{title[name]}",
        line=dict(color=color),
    )

    sample_dis_trace = go.Histogram(
        x=datasets[name][row][0].flatten(),
        nbinsx=100,
        histnorm="probability density",
        marker_color=color,
        showlegend=False,
    )

    sample_range_values = eeg_max_min[range_key[name]]["trail_range_values"]

    if "full" in name:
        sample_range_values = sample_range_values * 1e5

    sample_range_trace = go.Scatter(
        x=trail_len,
        y=sample_range_values,
        mode="lines",
        name=f"{title[name]}",
        line=dict(color=color),
    )

    sim_mat = cosine_similarity(datasets[name][row][0])
    ordered_similarity_matrix = reorder_similarity_matrix(sim_mat)
    np.fill_diagonal(ordered_similarity_matrix, -1)
    ordered_similarity_matrix_trace = go.Heatmap(
        z=ordered_similarity_matrix,
        zmin=-1,
        zmax=1,
        colorscale="Viridis",
    )

    np.fill_diagonal(sim_mat, -1)
    similarity_matrix_trace = go.Heatmap(
        z=sim_mat,
        zmin=-1,
        zmax=1,
        colorscale="Viridis",
        # name=str(int(np.sqrt(np.sum(sim_mat)))),
    )

    sim_mat_seq_traces = []
    submatrices = np.hsplit(data, seq_num)
    submatrices = (submatrices - np.mean(submatrices, axis=2, keepdims=True)) / np.std(
        submatrices, axis=2, keepdims=True
    )  # 标准化

    sim_sum_list = []
    for i, submatrix in enumerate(submatrices):

        sub_sim_mat = cosine_similarity(submatrix)
        sim_sum_list.append(np.sqrt(np.sum(sub_sim_mat)) / 96 * 100)
        np.fill_diagonal(sub_sim_mat, -1)
        sim_mat_ele_trace = go.Heatmap(
            z=sub_sim_mat,
            zmin=-1,
            zmax=1,
            colorscale="Viridis",
            showscale=False,
            name=str(int(sim_sum_list[i])),
        )
        sim_mat_seq_traces.append(sim_mat_ele_trace)
    # print(sim_sum_list)
    sim_mat_dis_trace = go.Histogram(
        x=sub_sim_mat.flatten(),
        nbinsx=100,
        histnorm="probability density",
        marker_color=color,
        name=f"{title[name]}",
    )

    return {
        "fig": fig_trace,
        "channel_range_fig": channel_range_trace,
        "sample_range_fig": sample_range_trace,
        "similarity_matrix_fig": [similarity_matrix_trace,ordered_similarity_matrix_trace],
        "sim_mat_seq_fig": sim_mat_seq_traces,
        "sim_mat_dis_fig": sim_mat_dis_trace,
        "sample_dis_fig": sample_dis_trace,
        "channel_dis_fig": channel_dis_trace,
    }


@app.route("/update_plot", methods=["POST"])
def update_plot():
    req_data = request.json
    row = int(req_data["row"])
    channel = int(req_data["channel"])
    selected_datasets = req_data["datasets"]
    label = datasets["raw_s1"][row][1]

    with ThreadPoolExecutor() as executor:
        futures = []
        for name in selected_datasets:
            if selected_datasets[name]:
                futures.append(
                    executor.submit(
                        create_figures,
                        row,
                        name,
                        colors[name],
                        channel,
                        datasets,
                    )
                )

        results = [future.result() for future in futures]

    response = build_response(results, label)

    return response


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
