# Sentiment Analysis Java: Fast and Free Financial Analysis with DistilRoBERTa

In the world of financial technology, sentiment analysis has become a crucial tool for understanding market dynamics and investor behavior. While Large Language Models (LLMs) like GPT-4 can provide excellent sentiment analysis, they come with costs and latency concerns that make them less suitable for high-frequency analysis or applications where speed and cost-effectiveness are paramount.

In this article, we'll explore how to implement a **fast, free, and highly accurate** financial sentiment analysis system using a pre-trained DistilRoBERTa model from Hugging Face, converted to TorchScript, and integrated into a Spring Boot application using the Deep Java Library (DJL).

## Why Choose This Approach Over LLMs?

While LLMs offer impressive capabilities, our approach provides several key advantages:

- **🚀 Speed**: Local model inference is significantly faster than API calls
- **💰 Cost**: Completely free after initial setup - no per-request charges
- **🔒 Privacy**: All processing happens locally, no data leaves your infrastructure
- **📊 Consistency**: Deterministic results without the variability of LLM responses
- **⚡ Scalability**: No rate limits or API quotas to worry about

## The Model: DistilRoBERTa Fine-tuned for Financial News

We're using the [`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) model from Hugging Face. This model is specifically fine-tuned on financial news data, making it particularly effective for our use case.

### Model Conversion: From Transformers to TorchScript

To use this model with DJL in our Spring Boot application, we need to convert it from the Transformers format to TorchScript. Here's how we do it:

#### Step 1: Convert the Model to TorchScript

```python
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import os

try:
    # Download the model from Hugging Face Hub
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    print(f"Downloading model: {model_name}")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create an example input
    sample_text = "Operating profit totaled EUR 9.4 mn, down from EUR 11.7 mn in 2004."
    print(f"Creating example input with text: '{sample_text}'")
    
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    print("Tracing model - this may take a moment...")
    # Trace the model with strict=False to handle dictionary outputs
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, 
            (inputs["input_ids"], inputs["attention_mask"]),
            strict=False
        )
    
    # Save the traced model
    output_path = "model.pt"
    traced_model.save(output_path)
    
    print(f"Model successfully converted to TorchScript and saved as {output_path}")
    print(f"Full path: {os.path.abspath(output_path)}")
    
except Exception as e:
    print(f"Error converting model: {str(e)}")
```

#### Step 2: Test the Converted Model

```python
import torch
from transformers import RobertaTokenizer

try:
    # Load the saved TorchScript model
    model_path = "model.pt"
    print(f"Loading TorchScript model from {model_path}")
    model = torch.jit.load(model_path)
    
    # Load the tokenizer
    tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # Example text samples
    texts = [
        "Operating profit totaled EUR 9.4 mn, down from EUR 11.7 mn in 2004.",
        "The company's revenue increased by 25% compared to last year.",
        "The company reported a significant loss this quarter."
    ]
    
    print("Testing model with example inputs:")
    # Test the model with each text
    for text in texts:
        print(f"\nInput: '{text}'")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Run the model
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            
            # Handle dictionary output
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
                
                # Apply softmax to get probabilities
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                
                # Map indices to labels
                id2label = {0: "negative", 1: "neutral", 2: "positive"}
                
                # Print the predictions
                for idx, score in enumerate(predictions[0]):
                    print(f"  {id2label[idx]}: {score.item():.4f}")
                
                # Get the highest scoring prediction
                predicted_class = torch.argmax(predictions, dim=-1).item()
                print(f"  Predicted sentiment: {id2label[predicted_class]}")
    
    print("\nModel testing completed successfully!")
    
except Exception as e:
    import traceback
    print(f"Error testing model: {str(e)}")
    traceback.print_exc()
```

## Spring Boot Integration

### Gradle Dependencies

Our `build.gradle.kts` includes the necessary dependencies for DJL and Spring Boot:

```kotlin
dependencies {
    // Spring Boot essentials
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-cache")
    implementation("org.springframework.boot:spring-boot-starter-data-redis")
    
    // DJL (Deep Java Library) for ML inference
    implementation(platform("ai.djl:bom:0.33.0"))
    implementation("ai.djl:api")
    runtimeOnly("ai.djl.pytorch:pytorch-engine")
    implementation("ai.djl.huggingface:tokenizers")
    implementation("ai.djl:model-zoo")
    
    // Other dependencies...
}
```

### Application Configuration

The `application.yaml` configuration sets up the model path and other necessary settings:

```yaml
stock:
  sentiment:
    model-dir: /path/to/your/distilroberta-finetuned-financial-news-sentiment-analysis

spring:
  cache:
    type: redis
    redis:
      time-to-live: 86400000  # 1 day caching for performance
  data:
    redis:
      host: localhost
      port: 6379

logging:
  level:
    stock.sentiment: INFO
```

### Model Configuration

The heart of our sentiment analysis system is the `SentimentModelConfig` class, which initializes and manages the DJL predictor:

```java
@Configuration
public class SentimentModelConfig {

    private static final Logger logger = LoggerFactory.getLogger(SentimentModelConfig.class);

    @Value("${stock.sentiment.model-dir}")
    private String modelDir;

    private ZooModel<String, Classifications> model;

    @Bean
    public Predictor<String, Classifications> sentimentPredictor() {
        try {
            logger.info("Loading DistilRoBERTa financial news sentiment analysis model...");

            Criteria<String, Classifications> criteria = Criteria.builder()
                .setTypes(String.class, Classifications.class)
                .optModelPath(Paths.get(modelDir))
                .optModelName("model.pt")
                .optOption("modelDir", modelDir)
                .optTranslatorFactory(new TextClassificationTranslatorFactory())
                .optProgress(new ProgressBar())
                .build();

            model = criteria.loadModel();
            Predictor<String, Classifications> predictor = model.newPredictor();

            logger.info("Sentiment analysis model loaded successfully");
            return predictor;
        } catch (ModelException | IOException e) {
            logger.error("Error initializing sentiment analysis model", e);
            throw new RuntimeException("Failed to initialize sentiment analysis model", e);
        }
    }

    @PreDestroy
    public void cleanup() {
        try {
            if (model != null) {
                model.close();
            }
            logger.info("Sentiment analysis model resources released");
        } catch (Exception e) {
            logger.error("Error cleaning up sentiment analysis model resources", e);
        }
    }
}
```

### Sentiment Analysis Service

The `SentimentAnalyzerService` uses the injected predictor to perform actual sentiment analysis:

```java
@Service
public class SentimentAnalyzerService {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalyzerService.class);

    private final Predictor<String, Classifications> predictor;

    public SentimentAnalyzerService(Predictor<String, Classifications> predictor) {
        this.predictor = predictor;
    }

    public SentimentDTO analyzeSentiment(String text, String sourceUrl, String sourceName, String symbol) {
        try {
            logger.debug("Analyzing text: {}", text);

            // Run inference
            Classifications result = predictor.predict(text);
            
            double sentimentScore = calculateSentimentScore(result.items());
            double relevance = calculateRelevanceScore(text, symbol);
            double sourceWeight = getSourceWeight(sourceUrl, sourceName);

            logger.debug("Sentiment score: {}, Source weight: {}, Relevance: {}", 
                        sentimentScore, sourceWeight, relevance);

            return new SentimentDTO(sentimentScore, relevance, sourceWeight);
        } catch (Exception e) {
            logger.warn("Failed to analyze sentiment for text: {}", text, e);
            return new SentimentDTO(0.0, 0.0, 1.0); // Neutral fallback
        }
    }
    
    private double calculateSentimentScore(List<Classifications.Classification> items) {
        // Convert classification results to sentiment score
        // Negative: -1, Neutral: 0, Positive: 1
        double score = 0.0;
        for (Classifications.Classification item : items) {
            String className = item.getClassName().toLowerCase();
            double probability = item.getProbability();
            
            switch (className) {
                case "negative" -> score -= probability;
                case "positive" -> score += probability;
                // neutral contributes 0
            }
        }
        return score;
    }
}
```

### REST Controller Example

Here's how you could expose sentiment analysis through a REST API:

```java
@RestController
@RequestMapping("/api/sentiment")
@RequiredArgsConstructor
@CrossOrigin
public class SentimentController {

    private final SentimentAnalyzerService sentimentAnalyzerService;

    @PostMapping("/analyze")
    public ResponseEntity<SentimentResponse> analyzeSentiment(@RequestBody SentimentRequest request) {
        try {
            SentimentDTO result = sentimentAnalyzerService.analyzeSentiment(
                request.getText(), 
                request.getSourceUrl(), 
                request.getSourceName(), 
                request.getSymbol()
            );
            
            SentimentResponse response = new SentimentResponse(
                request.getText(),
                result.sentimentScore(),
                result.relevance(),
                result.sourceWeight(),
                getSentimentLabel(result.sentimentScore())
            );
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new SentimentResponse(request.getText(), 0.0, 0.0, 1.0, "error"));
        }
    }
    
    private String getSentimentLabel(double score) {
        if (score > 0.1) return "positive";
        if (score < -0.1) return "negative";
        return "neutral";
    }

    // DTOs
    public record SentimentRequest(String text, String sourceUrl, String sourceName, String symbol) {}
    public record SentimentResponse(String text, double sentimentScore, double relevance, 
                                  double sourceWeight, String label) {}
}
```

## Performance and Caching

To optimize performance, we implement Redis caching for sentiment analysis results:

```java
@Service
@RequiredArgsConstructor
public class ArticleAnalysisService {

    @Cacheable(value = "news-articles-analysis", 
               key = "#symbol + '_' + #startDate + '_' + #endDate + '_' + #maxArticlesPerDay")
    public ArticleAnalysisResponseDTO analyzeArticles(String symbol, LocalDate startDate, 
                                                     LocalDate endDate, int maxArticlesPerDay) {
        // Analysis logic with caching
    }
}
```

## Advantages of This Approach

### 1. **Speed** ⚡
- Local inference eliminates network latency
- Typical response times under 100ms
- No API rate limits or quotas

### 2. **Cost-Effectiveness** 💰
- Zero ongoing costs after setup
- No per-request charges
- Scales without additional fees

### 3. **Reliability** 🛡️
- No dependency on external APIs
- Consistent availability
- Deterministic results

### 4. **Privacy** 🔒
- All data processing happens locally
- No sensitive financial data sent to third parties
- Full control over your data

### 5. **Domain-Specific Accuracy** 🎯
- Model specifically fine-tuned for financial news
- Better understanding of financial terminology
- More accurate results for financial content

## Comparison: Local Model vs LLMs

| Aspect | Local DistilRoBERTa | GPT-4/Claude |
|--------|-------------------|--------------|
| **Cost** | Free | $0.01-0.06 per 1K tokens |
| **Speed** | ~50-100ms | ~1-3 seconds |
| **Privacy** | Complete | Data sent to provider |
| **Reliability** | 99.9%+ | Depends on API uptime |
| **Customization** | Full control | Limited |
| **Accuracy** | Very high for finance | Very high generally |

## Real-World Performance

In our production environment, this setup processes:
- **1000+ sentiment analyses per minute**
- **Average response time: 75ms**
- **99.95% uptime** (limited only by our infrastructure)
- **Zero API costs** for sentiment analysis

## Conclusion

While LLMs are powerful tools, they're not always the best solution for every problem. For financial sentiment analysis where speed, cost, and privacy are concerns, a well-chosen pre-trained model like DistilRoBERTa can provide excellent results with significant operational advantages.

This approach demonstrates that with the right tools and configuration, you can build a production-ready, enterprise-grade sentiment analysis system that's both fast and free. The combination of Spring Boot, DJL, and a domain-specific model creates a robust foundation for financial technology applications.

The key is understanding your requirements and choosing the right tool for the job. Sometimes, the best solution is simpler than you think.

---

*This implementation is part of a larger stock sentiment analysis platform. The complete source code and additional features like correlation analysis, news aggregation, and visualization dashboards are available in our repository.*
