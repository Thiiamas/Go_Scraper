package main

import (
	"fmt"
	"github.com/gocolly/colly"
	"time"
	"strings"
	"regexp"
	"math"
	"os"
	"bufio"
	"log"
	"strconv"
)

const (
	service = "service"
	produit = "produit"
	cadre = "cadre"
	prix = "prix"
)

type wordFrequency struct {
	word string
	counter map[string]int 
}

type classifier struct{
	dataset map[string][]string 
	words map[string]wordFrequency
}

func main() {
	nb := newClassifier()
	//dataset from https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
	dataset := dataset("./Tripadvisor_tagged.txt")
	nb.train(dataset)
	//scraping
	var s = ScrapComment()
	
	for _,value := range s {
		result :=nb.classify(value)
		class := ""
		var proba []float64
		proba = append(proba,result[service])
		proba = append(proba,result[produit])
		proba = append(proba,result[cadre])
		proba = append(proba,result[prix])
		var final = 0.0
		var wtf = 0
		var wtf_save = 0
		for _,prob := range proba{
			if prob > final {
				final = prob
				wtf_save = wtf
			}
			wtf += 1
		}
		if wtf_save == 0 {
			class = service
		} else if wtf_save == 1{
			class = produit
		} else if wtf_save == 2{
			class = cadre
		} else if wtf_save == 3{
			class = prix
		}
		fmt.Printf("la phrase %v parle du %v\n",value,class)
	}
}

//Read a file and return it as a dataset struc
func dataset(file string) map[string]string {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	dataset := make(map[string]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		l := scanner.Text()
		data := strings.Split(l, "\t")
		if len(data) != 2 {
			continue
		}
		sentence := data[0]
		if data[1] == "0" {
			dataset[sentence] = service
		} else if data[1] == "1" {
			dataset[sentence] = produit
		} else if data[1] == "2" {
			dataset[sentence] = cadre
		} else if data[1] == "3" {
			dataset[sentence] = prix
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return dataset
}

//create a classifier
func newClassifier() *classifier {
	c := new(classifier)
	c.dataset = map[string][]string{
		service: []string{},
		produit: []string{},
		cadre: []string{},
		prix: []string{},
	}
	c.words = map[string]wordFrequency{}
	return c
	}

//train our classifier on a dataset
func (c *classifier) train(dataset map[string]string){
	for sentence, class := range dataset {
		c.addSentence(sentence,class)
		words := tokenize(sentence)
		for _,w := range words {
			c.addWord(w, class)
		}
	}
}	

func (c *classifier) addSentence(sentence, class string){
	c.dataset[class] = append(c.dataset[class], sentence)
}

func (c *classifier) addWord(word, class string) {
	wf, ok := c.words[word]
	if !ok {
			wf = wordFrequency{word: word, counter: map[string]int{
				service: 0,
				produit: 0,
				cadre: 0,
				prix: 0,
			}}
	}
	wf.counter[class]++
	c.words[word] = wf
			
}

//classify a sentence using Bayes
func (c classifier) classify(sentence string) map[string]float64{
	words := tokenize(sentence)
	serviceprob := c.probability(words, service)
	produitprob := c.probability(words, produit)
	cadreprob := c.probability(words, cadre)
	prixprob := c.probability(words, prix)
	return map[string]float64{
			service: serviceprob,
			produit: produitprob,
			cadre: cadreprob,
			prix: prixprob,
	}
	
}

//integration of Bayes methods

func(c classifier) priorProb(class string) float64{
	return float64(len(c.dataset[class]))/float64(len(c.dataset[service])+len(c.dataset[produit])+len(c.dataset[cadre])+len(c.dataset[prix]))
}

func (c classifier) totalWordCount(class string) int {
	serviceCount := 0
	produitCount := 0
	cadreCount := 0
	prixCount := 0
	for _, wf :=range c.words {
		serviceCount += wf.counter[service]
		produitCount += wf.counter[produit]
		cadreCount += wf.counter[cadre]
		prixCount += wf.counter[prix]
	}
	if class == service {
		return serviceCount
	}else if class == produit{
			return produitCount
	}else if class == cadre{
			return cadreCount
	}else if class == prix{
			return prixCount
	}else {
		return serviceCount + produitCount + cadreCount + prixCount
	}
}


func (c classifier) totalDistinctWordCount() int {
	serviceCount := 0
	produitCount := 0
	cadreCount := 0
	prixCount := 0
	for _, wf := range c.words {
		serviceCount += zeroOneTransform(wf.counter[service])
		produitCount += zeroOneTransform(wf.counter[produit])
		cadreCount += zeroOneTransform(wf.counter[cadre])
		prixCount += zeroOneTransform(wf.counter[prix])
	}
	return serviceCount + produitCount + cadreCount + prixCount
}

//Naive Bayes
func (c classifier) probability(words []string, class string)float64{
	prob := c.priorProb(class)
	for _, w := range words{
		count := 0
		if wf, ok := c.words[w]; ok {
				count = wf.counter[class]
		}
		prob *= (float64((count + 1)) / float64((c.totalWordCount(class) + c.totalDistinctWordCount())))
	}
	for _, w := range words {
		count := 0
		if wf, ok := c.words[w]; ok {
			count += (wf.counter[service] + wf.counter[produit]+wf.counter[cadre]+wf.counter[prix])
		}
		prob /= (float64((count +1)) / float64((c.totalWordCount("") + c.totalDistinctWordCount())))
	}
	return prob
	}

	
	

	


//UTILITiES
//
//
//

var stopwords = map[string]struct{}{
	"i": struct{}{}, "me": struct{}{}, "my": struct{}{}, "myself": struct{}{}, "we": struct{}{}, "our": struct{}{}, "ours": struct{}{},
	"ourselves": struct{}{}, "you": struct{}{}, "your": struct{}{}, "yours": struct{}{}, "yourself": struct{}{}, "yourselves": struct{}{},
	"he": struct{}{}, "him": struct{}{}, "his": struct{}{}, "himself": struct{}{}, "she": struct{}{}, "her": struct{}{}, "hers": struct{}{},
	"herself": struct{}{}, "it": struct{}{}, "its": struct{}{}, "itself": struct{}{}, "they": struct{}{}, "them": struct{}{}, "their": struct{}{},
	"theirs": struct{}{}, "themselves": struct{}{}, "what": struct{}{}, "which": struct{}{}, "who": struct{}{}, "whom": struct{}{}, "this": struct{}{},
	"that": struct{}{}, "these": struct{}{}, "those": struct{}{}, "am": struct{}{}, "is": struct{}{}, "are": struct{}{}, "was": struct{}{},
	"were": struct{}{}, "be": struct{}{}, "been": struct{}{}, "being": struct{}{}, "have": struct{}{}, "has": struct{}{}, "had": struct{}{},
	"having": struct{}{}, "do": struct{}{}, "does": struct{}{}, "did": struct{}{}, "doing": struct{}{}, "a": struct{}{}, "an": struct{}{},
	"the": struct{}{}, "and": struct{}{}, "but": struct{}{}, "if": struct{}{}, "or": struct{}{}, "because": struct{}{}, "as": struct{}{},
	"until": struct{}{}, "while": struct{}{}, "of": struct{}{}, "at": struct{}{}, "by": struct{}{}, "for": struct{}{}, "with": struct{}{},
	"about": struct{}{}, "against": struct{}{}, "between": struct{}{}, "into": struct{}{}, "through": struct{}{}, "during": struct{}{},
	"before": struct{}{}, "after": struct{}{}, "above": struct{}{}, "below": struct{}{}, "to": struct{}{}, "from": struct{}{}, "up": struct{}{},
	"down": struct{}{}, "in": struct{}{}, "out": struct{}{}, "on": struct{}{}, "off": struct{}{}, "over": struct{}{}, "under": struct{}{},
	"again": struct{}{}, "further": struct{}{}, "then": struct{}{}, "once": struct{}{}, "here": struct{}{}, "there": struct{}{}, "when": struct{}{},
	"where": struct{}{}, "why": struct{}{}, "how": struct{}{}, "all": struct{}{}, "any": struct{}{}, "both": struct{}{}, "each": struct{}{},
	"few": struct{}{}, "more": struct{}{}, "most": struct{}{}, "other": struct{}{}, "some": struct{}{}, "such": struct{}{}, "no": struct{}{},
	"nor": struct{}{}, "not": struct{}{}, "only": struct{}{}, "same": struct{}{}, "so": struct{}{}, "than": struct{}{}, "too": struct{}{},
	"very": struct{}{}, "can": struct{}{}, "will": struct{}{}, "just": struct{}{}, "don't": struct{}{}, "should": struct{}{}, "should've": struct{}{},
	"now": struct{}{}, "aren't": struct{}{}, "couldn't": struct{}{}, "didn't": struct{}{}, "doesn't": struct{}{}, "hasn't": struct{}{}, "haven't": struct{}{},
	"isn't": struct{}{}, "shouldn't": struct{}{}, "wasn't": struct{}{}, "weren't": struct{}{}, "won't": struct{}{}, "wouldn't": struct{}{},
}

//return True if w is in stopwords
func isStopWord(w string) bool {
	_, ok := stopwords[w]
	return ok
}

//lowercaize all character and remove unwanted ones
func cleanup(sentence string) string {
	re := regexp.MustCompile("[^a-zA-Z 0-9]+")
	return re.ReplaceAllString(strings.ToLower(sentence), "")
}

func tokenize (sentence string) []string {
	s := cleanup(sentence)
	words := strings.Fields(s)
	var tokens []string
	for _,w := range words{
		if !isStopWord(w){
			tokens = append(tokens,w)
		}
	}
	return tokens
}

// zeroOneTransform returns
//   0 if argument x = 0
//   1 otherwise
func zeroOneTransform(x int) int {
	return int(math.Ceil(float64(x) / (float64(x) + 1.0)))
}


//SCRAPING

func ScrapComment() []string{
		var s []string



	//collector
	c := colly.NewCollector(
	colly.UserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"),
	colly.AllowedDomains("www.tripadvisor.co.uk"),
	colly.AllowURLRevisit(),
	)
	
	//to avoid being banned...again
	c.Limit(&colly.LimitRule{
		DomainGlob : "www.allocine.fr/*",
		RandomDelay : 5 * time.Second,
	})

	//to print something when a new url is visited
	c.OnRequest(func(r *colly.Request) {
		fmt.Println("Visiting", r.URL.String())
	})

	//collect the Title of the comment
	c.OnHTML("span[class]", func(e *colly.HTMLElement) {
		if e.Attr("class") == "noQuotes"{
		//fmt.Println(e.Text)
		comment := strings.TrimSpace(e.Text)
		s = append(s,comment)
		}
		
	})
	
	//Next page
	page := 1
	c.OnHTML("a[data-page-number]", func(e *colly.HTMLElement){
		if e.Text == strconv.Itoa(page+1) {
			link := e.Attr("href")
			page +=1
			c.Visit(e.Request.AbsoluteURL(link))
		}
		
	})
	
	
	
	
	c.OnError(func(r *colly.Response, err error) {
		fmt.Println("Request URL:", r.Request.URL, "failed with response:", r, "\nError:", err)
	})
	
	//begin scrapping
	c.Visit("https://www.tripadvisor.co.uk/Restaurant_Review-g274707-d9863055-Reviews-LA_VIE_bistro_beef_shop-Prague_Bohemia.html#REVIEWS")
	
	return s
}