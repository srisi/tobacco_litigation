from tobacco_litigation.corpus import LitigationCorpus
import distance



def main(extract_size=100, max_dist=0.5):

    corpus = LitigationCorpus(load_part_of_speech=False)

    for doc_id1, doc1 in enumerate(corpus.get_search_term_extracts('defendant', 'quit',
                                                   extract_size=extract_size, no_passages=100000)):
        doc1t = doc1['text'].split()
        for doc_id2, doc2 in enumerate(corpus.get_search_term_extracts('defendant', 'quit',
                                                   extract_size=extract_size, no_passages=100000)):
            if doc_id1 == doc_id2 or doc_id1 >= doc_id2 or doc1['case'] == doc2['case']: continue
            doc2t = doc2['text'].split()
            dist = distance.nlevenshtein(doc1t, doc2t)
            if dist <= max_dist:

                print("\n\n", dist, doc_id1,
                      "\n", "{:30s}".format(doc1['case']), " ".join(doc1t),
                      "\n", "{:30s}".format(doc2['case']), " ".join(doc2t))


if __name__ == "__main__":
    main()