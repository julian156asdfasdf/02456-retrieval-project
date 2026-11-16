import hydra
from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.training.loss import DPRLoss
from src.training.trainer import Trainer
from src.evaluation.eval import evaluator

from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

@hydra.main(config_path='config', config_name='dpr_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]

    encoder = hydra.utils.instantiate(cfg.model)

    trainer = Trainer(encoder = encoder, lr = cfg.lr, batch_size= cfg.batch_size, epochs = cfg.epochs, save_path = cfg.save_dir)

    trainer.train(train_queries = train_queries, val_queries = val_queries, documents = corpus.all_documents, k = cfg.top_k)

    evaluate = evaluator(encoder = encoder, batch_size = cfg.batch_size, save_dir = cfg.save_dir)

    evaluate.eval(test_queries, documents = corpus.all_documents, k = cfg.top_k )




if __name__ == '__main__':
    main()