from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchvision.transforms import Normalize, Compose
from torch.utils.data import DataLoader

from dataset.dataloader import HDF5Dataset, CollatePad
from models.model import ImageFeatureExtractor
from models.transformer import Transformer
from metrics import Metrics
from utils.train import seed_everything, load_json
from utils.test import parse_arguments
from utils.utils import select_device

if __name__ == "__main__":
    args = parse_arguments()

    # Selecionar dispositivo (CPU ou GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"selected device is {device}.\n")

    # Carregar arquivo de configuração
    config = load_json(args.config_file_path)

    # Definir caminhos para o conjunto de dados e checkpoints
    dataset_dir = Path(args.dataset_directory)  # Diretório com arquivos mscoco hdf5 e json
    images_path = str(dataset_dir / "test_images.hdf5")
    captions_path = str(dataset_dir / "test_captions.json")
    lengths_path = str(dataset_dir / "test_lengthes.json")
    checkpoints_dir = config["paths"]["checkpoint"]
    checkpoint_name = args.checkpoint_path

    # Configurar semente para reprodutibilidade
    SEED = config["seed"]
    seed_everything(SEED)

    # Carregar vocabulário
    print("loading dataset...")
    vocab: Vocab = torch.load(str(dataset_dir / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    sos_id = vocab.stoi["<sos>"]
    eos_id = vocab.stoi["<eos>"]
    vocab_size = len(vocab)

    # Configurar DataLoader para testes
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    test_dataset = HDF5Dataset(hdf5_path=images_path,
                               captions_path=captions_path,
                               lengths_path=lengths_path,
                               pad_id=pad_id,
                               transform=transform)

    g = torch.Generator()
    g.manual_seed(SEED)
    max_len = config["max_len"]
    validation_loader = DataLoader(test_dataset,
                                   collate_fn=CollatePad(max_len, pad_id),
                                   batch_size=1,
                                   pin_memory=True,
                                   num_workers=4,
                                   shuffle=False)
    print("loading dataset finished.")

    # Construir modelos
    print("constructing models.\n")
    image_enc_config = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_config["encode_size"]**2)
    h, w = image_enc_config["encode_size"], image_enc_config[
        "encode_size"]

    transformer_config = config["hyperparams"]["transformer"]
    transformer_config["vocab_size"] = vocab_size
    transformer_config["pad_id"] = pad_id
    transformer_config["img_encode_size"] = image_seq_len
    transformer_config["max_len"] = max_len - 1

    image_encoder = ImageFeatureExtractor(**image_enc_config)
    transformer = Transformer(**transformer_config)

    # Carregar estados dos modelos
    load_path = str(Path(checkpoints_dir) / checkpoint_name)
    state = torch.load(load_path, map_location=torch.device("cpu"))
    image_encoder.load_state_dict(state["models"][0])
    transformer.load_state_dict(state["models"][1])

    image_encoder.to(device).eval()
    transformer.to(device).eval()

    # Avaliação do modelo
    eval_data = defaultdict(list)
    selected_data = defaultdict(list)
    metrics = Metrics()
    progress_bar = tqdm(validation_loader, leave=False, total=len(validation_loader))
    progress_bar.unit = "step"
    
    for imgs, cptns_all, lens in progress_bar:
        imgs: Tensor  # images [1, 3, 256, 256]
        cptns_all: Tensor  # all 5 captions [1, lm, cn=5]
        lens: Tensor  # lengthes of all captions [1, cn=5]

        k = 5
        # start: [1, 1]
        imgs = imgs.to(device)
        start = torch.full(size=(1, 1),
                           fill_value=sos_id,
                           dtype=torch.long,
                           device=device)
        with torch.no_grad():
            imgs_enc = image_encoder(imgs)  # [1, is, ie]
            logits, attns = transformer(imgs_enc, start)
            logits: Tensor  # [k=1, 1, vsc]
            attns: Tensor  # [ln, k=1, hn, S=1, is]
            log_prob = F.log_softmax(logits, dim=2)
            log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)
            # log_prob_topk [1, 1, k]
            # indices_topk [1, 1, k]
            current_preds = torch.cat(
                [start.expand(k, 1), indxs_topk.view(k, 1)], dim=1)
            # current_preds: [k, S]

            # get last layer, mean across transformer heads
            # attns = attns[-1].mean(dim=1).view(1, 1, h, w)  # [k=1, s=1, h, w]
            # current_attns = attns.repeat_interleave(repeats=k, dim=0)
            # [k, s=1, h, w]
        
        try:
            seq_preds = []
            seq_log_probs = []
            seq_attns = []
            while current_preds.size(1) <= (
                    max_len - 2) and k > 0 and current_preds.nelement():
                with torch.no_grad():
                    imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
                    # [k, is, ie]
                    logits, attns = transformer(imgs_expand, current_preds)
                    
                    # logits: [k, S, vsc]
                    # attns: # [ln, k, hn, S, is]
                    # get last layer, mean across transformer heads
                    # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                    # # [k, S, h, w]
                    # attns = attns[:, -1].view(k, 1, h, w)  # current word

                    # next word prediction
                    log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                    # log_prob: [k, vsc]
                    log_prob = log_prob + log_prob_topk.view(k, 1)
                    # top k probs in log_prob[k, vsc]
                    log_prob_topk, indxs_topk = log_prob.view(-1).topk(k,
                                                                       sorted=True)
                    
                    # indxs_topk are a flat indecies, convert them to 2d indecies:
                    # i.e the top k in all log_prob: get indecies: K, next_word_id
                    prev_seq_k, next_word_id = np.unravel_index(
                        indxs_topk.cpu(), log_prob.size())
                    next_word_id = torch.as_tensor(next_word_id).to(device).view(
                        k, 1)
                    # prev_seq_k [k], next_word_id [k]

                    current_preds = torch.cat(
                        (current_preds[prev_seq_k], next_word_id), dim=1)

                # find predicted sequences that ends
                seqs_end = (next_word_id == eos_id).view(-1)

                if torch.any(seqs_end):
                    seq_preds.extend(seq.tolist()
                                     for seq in current_preds[seqs_end])
                    seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                    # get last layer, mean across transformer heads
                    attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                    
                    # [k, S, h, w]
                    seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

                    k -= torch.sum(seqs_end)
                    current_preds = current_preds[~seqs_end]
                    log_prob_topk = log_prob_topk[~seqs_end]
                    # current_attns = current_attns[~seqs_end]

            # Sort predicted captions according to seq_log_probs
            specials = [pad_id, sos_id, eos_id]
            seq_preds, seq_attns, seq_log_probs = zip(*sorted(
                zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]))

            text_preds = [[vocab.itos[s] for s in seq if s not in specials]
                          for seq in seq_preds]
            text_refs = [[vocab.itos[r] for r in ref if r not in specials]
                         for ref in cptns_all.squeeze(0).permute(1, 0)]

            # calculate scores for each prediction
            scores = defaultdict(list)
            for text_pred in text_preds:
                for k, v in metrics.calculate([text_refs], [text_pred]).items():
                    scores[k].append(v)

            # save all eval data
            eval_data["hypos_text"].append(text_preds)
            eval_data["refs_text"].append(text_refs)
            eval_data["attns"].append(list(seq_attns))
            eval_data["log_prob"].append(list(seq_log_probs))
            for k, v_list in scores.items():
                eval_data[k].append(v_list)

            # save data for the prediction with the highest log_prob
            selected_data["hypos_text"].append(text_preds[0])
            selected_data["refs_text"].append(text_refs)
            selected_data["attns"].append(list(seq_attns)[0])
            selected_data["bleu1"].append(scores["bleu1"][0])
            selected_data["bleu2"].append(scores["bleu2"][0])
            selected_data["bleu3"].append(scores["bleu3"][0])
            selected_data["bleu4"].append(scores["bleu4"][0])
            selected_data["gleu"].append(scores["gleu"][0])
            selected_data["meteor"].append(scores["meteor"][0])

            # tracking some data on progress bar
            # show bleu4 for the prediction witthe highest log_prob

            progress_bar.set_description(
                f'bleu4: Current: {selected_data["bleu4"][-1]:.4f}, Max: {max(selected_data["bleu4"]):.4f}, Min: {min(selected_data["bleu4"]):.4f}, Mean: {mean(selected_data["bleu4"]):.4f} \u00B1 {pstdev(selected_data["bleu4"]):.2f}'  # noqa: E501
            )
        except Exception as e:
            print(e)
            print("entrou aqui")
            continue

    progress_bar.close()

    # Salvar resultados
    print("\nSaving data...")
    experiment_name = checkpoint_name.split("/")[0]
    save_dir = Path(args.output_directory) / f"{experiment_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data=eval_data).to_pickle(str(save_dir / "all.pickle"))
    pd.DataFrame(data=selected_data).to_pickle(str(save_dir / "selected.pickle"))

    print("Done.")
