# 魔法方法是python内置方法，不需要主动调用，存在的目的是为了给python的解释器进行调用。
# 几乎每个魔法方法都有一个对应的内置函数，或者运算符，当我们对这个对象使用这些函数或者运算符时就会调用类中的对应魔法方法。

# Mini-Batch GD中的重要概念：
# Epoch：对所有的Batch都进行一次前馈和反馈即为一个Epoch
# Batch-size：每个Batch的大小，即每个Batch含有样本的数量
# Iterations：Batch的数量。Batch-size * Iterations = 样本总数

# CLASS torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#                                   batch_sampler=None, num_workers=0, collate_fn=None,
#                                   pin_memory=False, drop_last=False, timeout=0,
#                                   worker_init_fn=None, multiprocessing_context=None,
#                                   generator=None, *, prefetch_factor=2, persistent_workers=False)
# dataset(Dataset): 传入的数据集
# batch_size(int, optional): 每个batch有多少个样本
# shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
# sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
# batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
# num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
# collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
# pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
# drop_last (bool, optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
# 如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。
# 一般只关注dataset,batch_size,shuffle,num_workers几个参数