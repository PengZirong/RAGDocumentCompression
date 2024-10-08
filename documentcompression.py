# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pylab as plt
from FlagEmbedding import FlagReranker

class OptimizedTextCompression:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reranker = FlagReranker(model_name, use_fp16=True)  # 初始化 reranker


    def compute_similarity(self, query, chunks):
        # 使用 reranker 计算相似度分数
        scores = self.reranker.compute_score([[query, chunk] for chunk in chunks], normalize=True)
        return scores

    def chunk_by_tokens(self, input_text: str, chunk_size: int = 20):
        tokens = self.tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
        token_offsets = tokens['offset_mapping']

        chunks = []
        span_annotations = []

        for i in range(0, len(token_offsets), chunk_size):
            chunk_end = min(i + chunk_size, len(token_offsets))
            if chunk_end - i > 0:
                start_offset = token_offsets[i][0]
                end_offset = token_offsets[chunk_end - 1][1]
                chunks.append(input_text[start_offset:end_offset])
                span_annotations.append((i, chunk_end))

        return chunks, span_annotations

    def calculate_log(self, similarities):
        similarities = np.clip(similarities, 1e-10, 1.-1e-10)
        focal_log = - (1 - similarities) ** 4 * np.log(similarities)
        print("calculate_log:")
        print(focal_log)
        print(sum(focal_log))
        plt.bar(range(len(focal_log)), focal_log)
        plt.ylim(min(focal_log) * 0.98, max(focal_log) * 1.02)
        plt.show()

        l = len(focal_log)
        best_start = 0
        best_end = l
        max_iterations=5
        tol=1e-5
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}")

            # 计算当前区间的均值和标准差
            current_focal_log = focal_log[best_start:best_end]
            mean = np.mean(current_focal_log)
            std = np.std(current_focal_log)
            threshold_stats = mean + 0.75 * std
            print(f"Mean: {mean}, Std: {std}, Threshold: {threshold_stats}")

            prev_best_start, prev_best_end = best_start, best_end

            # 重新确定 best_start
            new_best_start = best_start
            for i in range(best_start, best_end - 2):
                window1 = focal_log[i]
                window2 = np.mean(focal_log[i:i + 1])
                window3 = np.mean(focal_log[i:i + 2])
                if max(window1, window2, window3) > threshold_stats:
                    new_best_start = i + 1
                else:
                    break
            best_start = new_best_start

            # 重新确定 best_end
            new_best_end = best_end
            for i in range(1, best_end - best_start - 1):
                idx = best_end - i
                window1 = focal_log[idx]
                window2 = np.mean(focal_log[idx-1:idx])
                window3 = np.mean(focal_log[idx-2:idx])
                if max(window1, window2, window3) > threshold_stats:
                    new_best_end = idx
                else:
                    break
            best_end = new_best_end

            print(f"Updated best_start: {best_start}, best_end: {best_end}")

            # 检查是否收敛
            if (best_start == prev_best_start) and (best_end == prev_best_end):
                print("Converged.")
                break
        else:
            print("Reached maximum iterations without convergence.")

        # 最终绘制结果
        plt.bar(range(best_start, best_end), focal_log[best_start:best_end])
        plt.ylim(min(focal_log[best_start:best_end]) * 0.98, max(focal_log[best_start:best_end]) * 1.02)
        plt.title(f"Final Log from {best_start} to {best_end}")
        plt.show()

        print(best_start, best_end)
        return best_start, best_end

    def naive_chunking(self, chunks):
        outputs = []
        for chunk in chunks:
            outputs.append(self.encode(chunk))
        return outputs

    def optimize_boundaries(self, chunks, similarities, start, end):
        while start > 0 and similarities[start - 1] > similarities[start]:
            start -= 1
        while end < len(chunks) - 1 and similarities[end + 1] > similarities[end]:
            end += 1
        return start, end

    def compress_text(self, query, text, chunk_size=20):
        chunks, span_annotations = self.chunk_by_tokens(text, chunk_size)
        print(chunks)
        print("分片总数", len(chunks))

        # 使用 reranker 计算 chunk_embeddings 的相似度
        # similarities = [self.compute_similarity(query, chunk) for chunk in chunks]
        similarities = self.compute_similarity(query, chunks)
        plt.bar(range(len(similarities)), similarities)
        plt.ylim(min(similarities) * 0.98, max(similarities) * 1.02)
        plt.show()

        print("查询相似度", similarities)
        overall_similarity = np.mean(similarities)
        print("整体查询相似度", overall_similarity)

        best_start, best_end = self.calculate_log(similarities)
        print("最优边界：", best_start, best_end)
        best_start, best_end = self.optimize_boundaries(chunks, similarities, best_start, best_end)
        print("最优边界：", best_start, best_end)


        # 选择最佳的连续片段
        selected_chunks = chunks[best_start:best_end]

        compressed_text = ''.join(selected_chunks)

        compression_info = {
            'original_length': len(text),
            'compressed_length': len(compressed_text),
            'compression_ratio': len(compressed_text) / len(text),
            'overall_similarity': overall_similarity,
        }

        return compressed_text, compression_info

# 使用示例
compressor = OptimizedTextCompression(model_name='BAAI/bge-reranker-v2-m3')

query = "机器学习在自然语言处理中的应用"
text = """
机器学习是一种让计算机通过数据进行学习的技术。近年来，随着大数据和计算能力的提升，机器学习在各个领域取得了显著的成果。
自然语言处理是机器学习的一个重要应用领域。它涉及到计算机和人类语言之间的互动，如文本分析、语音识别和机器翻译等。
通过使用深度学习模型，尤其是神经网络，自然语言处理的性能得到了极大的提升。
例如，BERT模型在多个语言理解任务中表现出色，成为自然语言处理的一个重要里程碑。
除了文本分类和情感分析，机器学习还在自动摘要、问答系统和对话系统等方面展现了强大的能力。
尽管如此，机器学习在自然语言处理中的应用仍面临诸多挑战，如数据稀缺、多语言处理和模型的可解释性等。
未来，随着技术的不断进步，机器学习将进一步推动自然语言处理的发展，带来更多创新的应用。
"""


text += """二、非选择题阅读材料，完成13～14题。●材料一：《尚书·禹贡》号称夏朝的古文献，这一先秦文献将当时认识到的黄河与长江流域分为九个大的区域即“九州”，简要叙述了各区域的物产、土壤、山川及族群情况，其中冀、兖、青、徐、豫、雍六州属黄河流域，扬、荆、梁三州属长江流域。该文献根据适合农业与否，将各地土壤分为上上至下下九个等级，总体上黄河流域各区域更适合农业生产，并称以渭河平原为中心的雍州“厥土惟黄壤，厥田惟上上”，而今江浙一带的扬州，“厥土惟涂泥，厥田唯下下”。司马迁在《史记·货殖列传》中，分叙汉武帝时全国各地的社会经济与人文情况，说：“关中自?、雍以东至河、华，膏壤沃野千里，自虞夏之贡以为上田……故关中之地，于天下三分之一，而人众不过什三。然量其富，什居其六”；而长江中下游以南，“地广人稀，饭稻羹鱼，或火耕而水耨……江淮以南，无冻饿之人，亦无千金之家。 ”●材料二：两汉时期，长江流域社会经济发展迅速。比较《汉书》所记西汉元始二年(公元2年)及《续汉书·郡国志》所记东汉永和五年 (公元140年)的人口记录可知，在近一个半世纪中，全国总人口呈负增长，而长江流域各地区户数与口数都有不同程度的增长，甚至异常的增长。其中包括浙江、福建二省及江苏、安徽二省长江以南地区的会稽、丹阳、吴郡，户数增长28.11%，口数增长26.05%；与今江西省大体 相 当 的 豫 章 郡 户 数 增 长502.70%，口数增长374.16%；荆州长江以南的长沙、零陵、桂阳、武陵四郡大体上与今湖南省相当，户数增 长 达 412.25% """


compressed_text, compression_info = compressor.compress_text(query, text)

print("压缩后的文本:")
print(compressed_text)
print("\n压缩信息:")
print(compression_info)

