# Summarization Task - PyTorch

This is a Chinese news abstractive summarization model implemention.
The project is written in PyTorch(v1.6). I didn't use any tricky library/function, all vanilla numpy/pytorch module, so it should be ok under various PyTorch version.

Pretained weights/results release soon.



TODO:
- [x] Seq2Seq model(LSTM as basic block), training from scratch, w/ multi-head attention and prior knowledge, use beam search for decoding
- [ ] Use pretrained embedding instead of training from scratch
- [ ] top-k/top-p decoding
- [ ] GPT-2 w/ improved mask
- [ ] train w/ r-drop
- [ ] Extractive Summarization w/ BERT
