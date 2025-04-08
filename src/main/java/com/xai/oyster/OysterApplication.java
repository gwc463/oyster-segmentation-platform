package com.xai.oyster;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
public class OysterApplication {
    public static void main(String[] args) {
        SpringApplication.run(OysterApplication.class, args);
    }
}

@RestController
class TestController {
    @GetMapping("/test")
    public String test() {
        return "Hello, Oyster!";
    }
}