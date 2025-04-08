package com.xai.oyster.controller;

import com.xai.oyster.entity.User;
import com.xai.oyster.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
public class UserController {
    @Autowired
    private UserService userService;

    @PostMapping("/login")
    public User login(@RequestParam String username, @RequestParam String password) {
        return userService.login(username, password);
    }
}